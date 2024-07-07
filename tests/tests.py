import torch
import time

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch

# Dummy function to create one-hot encoded tensor using loop
def one_hot_encoding_loop(src_logits_shape, target_classes_o, batch_idx, src_idx):
    target_classes_onehot = torch.zeros(src_logits_shape, dtype=target_classes_o.dtype, layout=target_classes_o.layout, device=target_classes_o.device)
    for b_idx, s_idx in zip(batch_idx, src_idx):
        target_classes_onehot[b_idx, s_idx] = target_classes_o[b_idx]
    return target_classes_onehot

# Dummy function to create one-hot encoded tensor using vectorized assignment
def one_hot_encoding_vectorized(src_logits_shape, target_classes_o, batch_idx, src_idx):
    target_classes_onehot = torch.zeros(src_logits_shape, dtype=target_classes_o.dtype, layout=target_classes_o.layout, device=target_classes_o.device)
    target_classes_onehot[(batch_idx, src_idx)] = target_classes_o
    return target_classes_onehot

# Test function to compare both implementations
def test_one_hot_encoding():
    # Create dummy predictions and targets
    src_logits_shape = (2, 5, 3)  # batch_size=2, num_queries=5, num_classes=3
    target_classes_o = torch.tensor([
        [1, 0, 1],  # first batch, first target
        [0, 1, 0],  # first batch, second target
        [1, 1, 1],  # first batch, third target
        [0, 1, 1],  # second batch, first target
        [1, 0, 0]   # second batch, second target
    ], dtype=torch.float32, device='cpu')
    
    # Dummy indices matching preds to targets
    batch_idx = torch.tensor([0, 0, 0, 1, 1])
    src_idx = torch.tensor([0, 1, 2, 0, 2])

    # Compute one-hot encodings with both implementations
    onehot_loop = one_hot_encoding_loop(src_logits_shape, target_classes_o, batch_idx, src_idx)
    onehot_vectorized = one_hot_encoding_vectorized(src_logits_shape, target_classes_o, batch_idx, src_idx)

    # Print results for comparison
    print("One-hot encoding (Loop-based):\n", onehot_loop)
    print("One-hot encoding (Vectorized):\n", onehot_vectorized)

    # Assert that both implementations produce the same results
    assert torch.allclose(onehot_loop, onehot_vectorized), "Loop-based one-hot encoding does not match vectorized one-hot encoding!"



def test_classification_cost():
    # Create dummy data
    num_classes = 10
    num_predictions = 500
    num_targets = 20
    out_prob = torch.rand(num_predictions, num_classes) # bs, num_queries, num_classes
    
    tgt_ids = torch.zeros(num_targets, num_classes)
    one_indices = torch.randint(0, num_classes, (num_targets,))
    tgt_ids[torch.arange(num_targets), one_indices] = 1
    
    # Define alpha and gamma for focal loss
    alpha = 0.25
    gamma = 2.0

    # Compute neg_cost_class and pos_cost_class
    neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
    pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())

    # Vectorized implementation
    timenow = time.time()
    N, C = out_prob.shape
    M = tgt_ids.shape[0]
    tgt_labels_expanded = tgt_ids.unsqueeze(0).expand(N, M, C)
    pos_cost_class_expanded = pos_cost_class.unsqueeze(1).expand(N, M, C)
    neg_cost_class_expanded = neg_cost_class.unsqueeze(1).expand(N, M, C)
    # Calculate the cost class matrix
    cost_class_vectorized = (tgt_labels_expanded * pos_cost_class_expanded - tgt_labels_expanded * neg_cost_class_expanded).sum(-1)
    print("vectorized takes: ", time.time()-timenow)

    timenow = time.time() 
    # Loop implementation
    cost_class_loop = torch.zeros((N, M))
    for i in range(N):
        for j in range(M):
            classes = tgt_ids[j]
            one_indices = torch.nonzero(classes, as_tuple=False)[0]
            for cidx in one_indices:
                pos_cost = pos_cost_class[i, cidx]
                neg_cost = neg_cost_class[i, cidx]
                cost_class_loop[i, j] += pos_cost - neg_cost
    
    print("loop takes: ", time.time()-timenow)
    # Print results
    # print("Cost Class (Vectorized):\n", cost_class_vectorized)
    # print("Cost Class (Loop):\n", cost_class_loop)

    # Check if both implementations give the same result
    assert torch.allclose(cost_class_vectorized, cost_class_loop), "The vectorized and loop implementations do not match!"
    print("Hurrah!, test passed!")


def forward_embed_loop(known_labels_expand, label_enc, hidden_dim):
    if known_labels_expand.dim() == 2:
        return label_enc(known_labels_expand)

    bs, num_instances, num_classes = known_labels_expand.shape

    # Initialize a tensor to hold the final embeddings for each label
    aggregated_embeddings = torch.zeros(bs, num_instances, hidden_dim, device=known_labels_expand.device)

    class_mask = known_labels_expand.unsqueeze(-1)
    all_indices = torch.nonzero(class_mask == 1, as_tuple=False)[:, :3]
    for i in range(bs):
        batch_indices = all_indices[:, 0] == i
        instance_indices = all_indices[batch_indices, 1]
        class_indices = all_indices[batch_indices, 2]
        class_embeddings = label_enc(class_indices)
        # add these class_embeddings to the corresponding instance indices given by all_indices[:, 1]
        aggregated_embeddings[i].scatter_add_(0, instance_indices.unsqueeze(-1).expand_as(class_embeddings), class_embeddings)

    return aggregated_embeddings


def forward_embed_nested_loops(known_labels_expand, label_enc, hidden_dim):
    if known_labels_expand.dim() == 2:
        return label_enc(known_labels_expand)

    bs, num_instances, num_classes = known_labels_expand.shape

    # Initialize a tensor to hold the final embeddings for each label
    aggregated_embeddings = torch.zeros(bs, num_instances, hidden_dim, device=known_labels_expand.device)

    for b in range(bs):
        for instance in range(num_instances):
            one_indices = torch.nonzero(known_labels_expand[b, instance], as_tuple=False)[0]
            for clas_idx in one_indices:
                aggregated_embeddings[b, instance] += label_enc(clas_idx)

    return aggregated_embeddings


def test_embedding_forward():
    
    num_classes = 17
    hidden_dim = 56
    batch_size = 2
    num_instances = 10
    known_labels_expand = torch.zeros(batch_size * num_instances, num_classes)
    class_indices = torch.randint(0, num_classes, (batch_size * num_instances,))
    known_labels_expand[torch.arange(batch_size * num_instances), class_indices] = 1
    known_labels_expand = known_labels_expand.view(batch_size, num_instances, num_classes)

    # Set the embedding weights to the same fixed values for all models
    label_enc = nn.Embedding(num_classes + 1, hidden_dim)

    # Initialize models
    output_loop = forward_embed_loop(known_labels_expand, label_enc, hidden_dim)
    output_vectorized = forward_embed_vectorized(known_labels_expand, label_enc, hidden_dim)
    # output_nested_loops = forward_embed_nested_loops(known_labels_expand, label_enc, hidden_dim)

    # Create dummy input


    # Print results for comparison
    print("Output (Loop-based):\n", output_loop)
    print("Output (Vectorized):\n", output_vectorized)
    # print("Output (Nested Loops):\n", output_nested_loops)

    # Assert that all outputs are the same
    # assert torch.allclose(output_loop, output_nested_loops), "Loop-based output does not match nested loops output!"
    # assert torch.allclose(output_loop, output_vectorized), "Loop-based output does not match vectorized output!"



if __name__ == "__main__":
    # Call the test function
    # test_classification_cost()
    # Run the test function
    # test_one_hot_encoding()
    test_embedding_forward()
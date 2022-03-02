def min_cost_matching(distance_metric, max_distance, tracks, detections, track_indices = None, detection_indices = None):
    # compute the const matrix
    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)

    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    # hungarian A, index: row--> tracks, columns --> detections
    row_indices, col_indices = linear_assignment(cost_matrix)
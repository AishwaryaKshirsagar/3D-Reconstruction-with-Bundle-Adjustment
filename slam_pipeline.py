import numpy as np 
import json
class SlamPipeline:
    def get_fund_mat(self, correspondences):
        system_matrix = []

        for correspondence in correspondences:
            [x1, y1], [x2, y2] = correspondence
            system_matrix.append([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])

        corr_mat = np.array(system_matrix)
        U, S, Vt = np.linalg.svd(corr_mat)
        # Set last singular value to zero to enforce rank 2 in the matrix and therefore have epipoles
        if len(S) == 9:
            S[-1] = 0
            updated_corr_map = np.dot(U[:, :len(S)] * S, Vt)
            U, S, Vt = np.linalg.svd(updated_corr_map)
        # Last column of V contains F. Since we have Vt, we keep last row instead.
        F_row = Vt[-1, :]
        F = F_row.reshape(3, 3)
        F /= F[2, 2]
        return F

    def get_ransac_fund_mat(self, correspondences, num_iterations, threshold):
        best_F = None
        best_inliers = []

        for _ in range(num_iterations):
            # Randomly sample 8 correspondences
            sample_indices = np.random.choice(len(correspondences), 8, replace=False)
            sample_correspondences = [correspondences[i] for i in sample_indices]

            # Estimate fundamental matrix using the sampled correspondences
            F_estimate = self.get_fund_mat(sample_correspondences)

            # Calculate the geometric error for each correspondence
            inliers = []
            for i, correspondence in enumerate(correspondences):
                x1, y1 = correspondence[0]
                x2, y2 = correspondence[1]

                line1 = np.dot(F_estimate, [x1, y1, 1])
                line2 =  np.dot(F_estimate, [x2, y2, 1])
                error = np.abs(np.dot(line2, np.cross(line1, line2))) /  np.sqrt(np.sum(line1[:2]**2) * np.sum(line2[:2]**2)) * 10**17
                if error < threshold:
                    inliers.append(i)

            # Keep track of the best set of inliers
            if len(inliers) > len(best_inliers):
                best_inliers = inliers

        # Refit the fundamental matrix using all inliers
        inlier_correspondences = [correspondences[i] for i in best_inliers]
        best_F = self.get_fund_mat(inlier_correspondences)

        return best_F, len(inlier_correspondences)
    
    def get_ess_mat(self, F, params_path):
        # Load camera parameters file
        json_file_path = params_path
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

        # Access the intrinsic matrices for each image
        intrinsics_data = data.get('intrinsics', {})

        K = intrinsics_data

        # SVD 
        essential_matrix = np.dot(np.dot(np.transpose(K), F), K)
        U, S, Vt = np.linalg.svd(essential_matrix)

        singular_values = np.array([1, 1, 0])
        E = np.dot(np.dot(U, np.diag(singular_values)), Vt)
        return E



"""_summary_

    Returns:
        _type_: _description_
"""
import time
import math
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances


start = time.time()

SIZE = 22


class CalculateComplexity():
    """_summary_
    """

    def get_input_matrix(self, filepath):
        """_summary_

        Args:
            filepath (_type_): _description_

        Returns:
            _type_: _description_
        """

        input_data = pd.read_excel(filepath, header=None).squeeze(1)
        input_data = input_data.values
        return input_data

    def find_rank_matrix_faster(self, input_data):
        '''
        upper triangular elements of matrix are taken and sorted. The indices
         of these elements in the sorted array + 1 is
        the required rank. Also required that duplicate elements should not be
         given the same rank. So a hashmap called index_map
        is used where the key is the element in the matrix and the value is
         the index of the last occurence of the element in the
        sorted array.

        Now every element in the uppertriangular part of the matrix is
         traversed and using that as a key, the rank is calculated
        from the index_map. rank = index + 1. Now check is made to see if
         there was a duplicate element in the sorted array just
        before the index (index-1) , if so index_map[element] is assigned
         index-1, so that next encounter of the element in the
        matrix will get a different rank.
        '''
        upper_triangular_elements = input_data[np.triu_indices(SIZE, k=0)]
        upper_triangular_elements_sorted = sorted(
            upper_triangular_elements, reverse=True)
        index_map = {upper_triangular_elements_sorted[i]: i for i in range(
            len(upper_triangular_elements_sorted))}

        #   print(upper_triangular_elements_sorted)
        rank_matrix = np.zeros((SIZE, SIZE), np.uint8)

        #   filling the rank_matrix with ranks
        for i in range(SIZE):
            for j in range(i+1, SIZE):  # upper triangle
                index = index_map[input_data[i][j]]
                rank_matrix[i][j] = index + 1  # ranks start from 1 and not 0
                rank_matrix[j][i] = index + 1

                if (index > 0 and upper_triangular_elements_sorted[index-1]
                        == upper_triangular_elements_sorted[index]):
                    index_map[input_data[i][j]] = index-1

        return rank_matrix

    def get_random_2d_points(self, number_of_points):
        """_summary_

        Args:
            number_of_points (_type_): _description_

        Returns:
            _type_: _description_
        """
        # MAX_VALUE = 1 # max value for x and y
        # list of random 2d points
        # points = np.array(list(zip(np.random.randint(MAX_VALUE,
        # size=number_of_points),np.random.randint(MAX_VALUE,size=number_of_points))))
        list1 = []
        list2 = []

        def merge(list1, list2):

            merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
            return merged_list

        # Driver code
        # list1 = [1, 2, 3]
        # list2 = ['a', 'b', 'c']
        # print(merge(list1, list2))

        for i in range(1, SIZE+1):
            x = (i-1)/(SIZE-1)
            #        y = ((1/2)*(1+math.sin(2*(math.pi)*x - math.pi)))
            y = ((1/2)*(1-math.sin(2*(math.pi)*x)))
            #  y = ((1/2)*(1+math.sin((math.pi)*x)))
            #  y = ((1/2)*(1+math.cos(2*(math.pi)*x - math.pi)))
            #  y = ((1/2)*(1+math.cos(2*(math.pi)*x)))
            #  y = ((1/2)*(1+math.cos((math.pi)*x)))
            #  y = ((1/2)*(1+math.cot(2*(math.pi)*x - math.pi)))
            list1.append(x)
            list2.append(y)
            points = np.array(merge(list1, list2))
        return points

    def get_distance_matrix(self, points):
        """_summary_

        Args:
            points (_type_): _description_

        Returns:
            _type_: _description_
        """
        # eucledian distance matrix from points
        distance_matrix = euclidean_distances(points)
        return distance_matrix

    def get_corrected_distance_matrix(self, distance_matrix,
                                      similarity_matrix_rank):
        """_summary_

        Args:
            distance_matrix (_type_): _description_
            similarity_matrix_rank (_type_): _description_
        """
        upper_triangular_elements = distance_matrix[np.triu_indices(SIZE, k=0)]
        # print('Distance matrix',distance_matrix)
        # print('Upper elems',upper_triangular_elements)
        upper_triangular_elements_sorted = sorted(
            upper_triangular_elements, reverse=True)
        # print('Upper elems sorted',upper_triangular_elements_sorted)
        new_distance_matrix = np.zeros((SIZE, SIZE))

        for i in range(SIZE):
            for j in range(SIZE):
                if i != j:
                    new_distance_matrix[i][j] = upper_triangular_elements_sorted[similarity_matrix_rank[i][j] - 1]
                    new_distance_matrix[j][i] = upper_triangular_elements_sorted[similarity_matrix_rank[j][i] - 1]

        return new_distance_matrix

    def calculate_fmatrix(self, new_distance_matrix, distance_matrix):
        """_summary_

        Args:
            new_distance_matrix (_type_): _description_
            distance_matrix (_type_): _description_
        """
        #     f_matrix = 1 - np.divide(new_distance_matrix,distance_matrix)
        # #     print(f_matrix)
        #     np.fill_diagonal(f_matrix,0)  # remove -inf along the diagonals
        # #     print(f_matrix)

        f_matrix = np.zeros((SIZE, SIZE))

        for i in range(SIZE):
            for j in range(SIZE):
                if i != j:
                    f_matrix[i][j] = 1 - \
                        (new_distance_matrix[i][j]/distance_matrix[i][j])

        return f_matrix

    def calculate_new_points(self, points, f_matrix):
        """_summary_

        Args:
            points (_type_): _description_
            f_matrix (_type_): _description_

        Returns:
            _type_: _description_
        """
        new_points = []
        points = np.array(points)
        for i in range(len(points)):
            points_diff = points - points[i]
            f_elems = list(zip(f_matrix[i], f_matrix[i]))
            added_elems = np.sum(np.multiply(points_diff, f_elems), axis=0)
            new_pt = points[i] + 1.0/(SIZE-1) * added_elems
            new_points.append(list(new_pt))
        return new_points

    def check_convergence(self, distance_matrix, new_distance_matrix,
                          threshold):
        """_summary_

        Args:
            distance_matrix (_type_): _description_
            new_distance_matrix (_type_): _description_
            threshold (_type_): _description_

        Returns:
            _type_: _description_
        """
        diff_matrix = distance_matrix - new_distance_matrix

        for i in range(SIZE):
            for j in range(SIZE):
                # print('Checking Elem : ',i,j)

                if i != j and abs(diff_matrix[i][j]) >= threshold:
                    # print('index :: ',i,j)
                    # print('value :: ',diff_matrix[i][j])
                    return False

        return True

    def centroid(self, *points):
        """_summary_
        """
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        centroid_x = sum(x_coords)/SIZE
        centroid_y = sum(y_coords)/SIZE
        return [centroid_x, centroid_y]

    def complexity_metric(self, points):
        """_summary_

        Args:
            points (_type_): _description_

        Returns:
            _type_: _description_
        """
        # points = np.array([np.array([0,0.5]),np.array([0.333,0.0670]),
        # np.array([0.6667,0.9330]),np.array([1,0.5])])
        x = points[:, 0]
        y = points[:, 1]
        sig_x = np.std(x)
        sig_y = np.std(y)
        complexity_metric = np.sqrt(np.square(sig_x) + np.square(sig_y))
        # print(complexity_metric)
        return complexity_metric

    def calculate_shape_complexity(self, filepath):
        """_summary_
        """
        similarity_matrix = self.get_input_matrix(filepath)
        similarity_matrix_rank = self.find_rank_matrix_faster(
            similarity_matrix)
        points = np.array(self.get_random_2d_points(number_of_points=SIZE))

        while True:
            distance_matrix = self.get_distance_matrix(points)

            distance_matrix_rank = self.find_rank_matrix_faster(
                distance_matrix)
            new_distance_matrix = self.get_corrected_distance_matrix(
                distance_matrix, similarity_matrix_rank)

            f_matrix = self.calculate_fmatrix(
                new_distance_matrix, distance_matrix)

            converge = self.check_convergence(
                distance_matrix, new_distance_matrix, 0.001)

            if converge:
                rank = self.find_rank_matrix_faster(new_distance_matrix)
                break

            points = np.array(self.calculate_new_points(points, f_matrix))
            cent = self.centroid(*points)
            distmax = np.max(new_distance_matrix)
            simmax = np.max(similarity_matrix)
            sf = simmax/distmax
            scaledpoints = cent + sf * (points-cent)
            comp = self.complexity_metric(points)
            CompScaled = self.complexity_metric(scaledpoints)
        return comp

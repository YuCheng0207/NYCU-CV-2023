import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

image_row = 120
image_col = 120

# visualizing the mask (size : "image width" * "image height")


def mask_visualization(M):
    mask = np.copy(np.reshape(M, (image_row, image_col)))
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')

# visualizing the unit normal vector in RGB color space
# N is the normal map which contains the "unit normal vector" of all pixels (size : "image width" * "image height" * 3)


def normal_visualization(N):
    # converting the array shape to (w*h) * 3 , every row is a normal vetor of one pixel
    N_map = np.copy(np.reshape(N, (image_row, image_col, 3)))
    # Rescale to [0,1] float number
    N_map = (N_map + 1.0) / 2.0
    plt.figure()
    plt.imshow(N_map)
    plt.title('Normal map')

# visualizing the depth on 2D image
# D is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")


def depth_visualization(D):
    D_map = np.copy(np.reshape(D, (image_row, image_col)))
    # D = np.uint8(D)
    plt.figure()
    plt.imshow(D_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

# convert depth map to point cloud and save it to ply file
# Z is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")


def save_ply(Z, filepath):
    Z_map = np.reshape(Z, (image_row, image_col)).copy()
    data = np.zeros((image_row*image_col, 3), dtype=np.float32)
    # let all point float on a base plane
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val
    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[image_row - 1 - i][j]
    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(filepath, pcd, write_ascii=True)

# show the result of saved ply file


def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

# read the .bmp file


def read_bmp(filepath):
    global image_row
    global image_col
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    image_row, image_col = image.shape
    return image


def read_all_bmp(name):
    image_list = []
    path = './test/' + name + '/pic'
    for i in range(1, 7):
        image = read_bmp(path+str(i)+'.bmp')
        image = np.reshape(image, image_row*image_col)  # fixed size
        image_list.append(image)

    image_matrix = np.matrix(image_list)  # do matrix multplie
    return image_matrix


def read_LightSource_and_Norm(filepath):
    lightSource_list = []
    with open(filepath, 'r', encoding='utf8') as f:
        for line in f.readlines():
            pic = line[7:-2].split(',')
            pic = [float(i) for i in pic]
            lightSource_list.append(pic/np.linalg.norm(pic))

    lightSource_matrix = np.matrix(lightSource_list)  # do matrix multplie

    return lightSource_matrix


def normal_estimation(name):
    I = read_all_bmp(name)
    L = read_LightSource_and_Norm('./test/'+name+'/LightSource.txt')
    # formula KdN = (L^T*L)^-1 * L^T * I size (3, ?)
    KdN = (L.T * L).I * L.T * I
    KdN_norm = np.linalg.norm(KdN, axis=0)  # size of norm (?, )
    for idx in range(KdN_norm.shape[0]):  # replace 0 to 1e-6
        if KdN_norm[idx] == 0:
            KdN_norm[idx] = 1e-6

    KdN = np.array(KdN, dtype=float)
    N = KdN / KdN_norm
    N = N.T
    N = N.reshape(image_row, image_col, 3)
    normal_visualization(N)
    plt.show()

    return N


def from_top_left(N):
    top_left = np.zeros((image_row, image_col), dtype=float)
    x_gradient = np.zeros((image_row, image_col), dtype=float)
    y_gradient = np.zeros((image_row, image_col), dtype=float)
    for i in range(image_row):
        for j in range(image_col):
            if N[i][j][2] != 0:
                x_gradient[i][j] = -N[i][j][0] / N[i][j][2]
                y_gradient[i][j] = N[i][j][1] / N[i][j][2]

    for i in range(image_row):
        for j in range(image_col):
            if (i == 0 and j == 0) or (x_gradient[i][j] == 0 and y_gradient[i][j] == 0):
                top_left[i][j] = 0
            else:
                top_left[i][j] = (top_left[i-1][j] + top_left[i][j-1] +
                                  x_gradient[i][j] + y_gradient[i][j]) / 2

    return top_left


def from_top_right(N):
    top_right = np.zeros((image_row, image_col), dtype=float)
    x_gradient = np.zeros((image_row, image_col), dtype=float)
    y_gradient = np.zeros((image_row, image_col), dtype=float)
    for i in range(image_row-1, -1, -1):
        for j in range(image_col):
            if N[i][j][2] != 0:
                x_gradient[i][j] = -N[i][j][0] / N[i][j][2]
                y_gradient[i][j] = -N[i][j][1] / N[i][j][2]

    for i in range(image_row-1, -1, -1):
        for j in range(image_col):
            if (i == image_row-1 and j == 0) or (x_gradient[i][j] == 0 and y_gradient[i][j] == 0):
                top_right[i][j] = 0
            else:
                top_right[i][j] = (top_right[i+1][j] + top_right[i][j-1] +
                                   x_gradient[i][j] + y_gradient[i][j]) / 2

    return top_right


def from_bottom_right(N):
    bottom_right = np.zeros((image_row, image_col), dtype=float)
    x_gradient = np.zeros((image_row, image_col), dtype=float)
    y_gradient = np.zeros((image_row, image_col), dtype=float)
    for i in range(image_row-1, -1, -1):
        for j in range(image_col-1, -1, -1):
            if N[i][j][2] != 0:
                x_gradient[i][j] = N[i][j][0] / N[i][j][2]
                y_gradient[i][j] = -N[i][j][1] / N[i][j][2]

    for i in range(image_row-1, -1, -1):
        for j in range(image_col-1, -1, -1):
            if (i == image_row-1 and j == image_col-1) or (x_gradient[i][j] == 0 and y_gradient[i][j] == 0):
                bottom_right[i][j] = 0
            else:
                bottom_right[i][j] = (bottom_right[i+1][j] + bottom_right[i][j+1] +
                                      x_gradient[i][j] + y_gradient[i][j]) / 2

    return bottom_right


def from_bottom_left(N):
    bottom_left = np.zeros((image_row, image_col), dtype=float)
    x_gradient = np.zeros((image_row, image_col), dtype=float)
    y_gradient = np.zeros((image_row, image_col), dtype=float)
    for i in range(image_row):
        for j in range(image_col-1, -1, -1):
            if N[i][j][2] != 0:
                x_gradient[i][j] = N[i][j][0] / N[i][j][2]
                y_gradient[i][j] = N[i][j][1] / N[i][j][2]

    for i in range(image_row):
        for j in range(image_col-1, -1, -1):
            if (i == 0 and j == image_col-1) or (x_gradient[i][j] == 0 and y_gradient[i][j] == 0):
                bottom_left[i][j] = 0
            else:
                bottom_left[i][j] = (bottom_left[i-1][j] + bottom_left[i][j+1] +
                                     x_gradient[i][j] + y_gradient[i][j]) / 2

    return bottom_left


def surface_reconstruction1(N):
    top_left = from_top_left(N)
    top_right = from_top_right(N)
    bottom_right = from_bottom_right(N)
    bottom_left = from_bottom_left(N)

    Z = (top_left + top_right + bottom_right + bottom_left) / 4

    return Z


def detect_extreme(Z):
    threshold = 50
    neighbors = np.zeros((3, 3), dtype=float)
    for i in range(1, image_row-1):
        for j in range(1, image_col-1):
            neighbors = Z[i-1:i+2, j-1:j+2]
            if (Z[i][j] >= threshold):
                Z[i][j] = (neighbors[0][0] + neighbors[0][2] +
                           neighbors[2][0] + neighbors[2][2]) / 4

    return Z


if __name__ == '__main__':
    images = ['bunny', 'star', 'venus']
    for name in images:
        N = normal_estimation(name)
        Z = surface_reconstruction1(N)
        if name == 'venus':
            Z = detect_extreme(Z)
        depth_visualization(Z)
        plt.show()

        filepath =  name + '.ply'
        save_ply(Z, filepath)
        show_ply(filepath)

    # showing the windows of all visualization function
    # plt.show()

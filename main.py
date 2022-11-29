"""
    This is the main file of stl_viewer backend module.
"""
import os
import math
import time
import trimesh
from fastapi import FastAPI, UploadFile
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from matplotlib import pyplot as plt
from descartes import PolygonPatch
import numpy as np
import cv2
import xlsxwriter
from CalculateComplexity import CalculateComplexity

origins = [
    "http://localhost",
    "http://localhost:4001",
]

app = FastAPI()
calc = CalculateComplexity()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


DESTINATION = "/"
CHUNK_SIZE = 2 ** 20  # 1MB


async def chunked_copy(src, dst):
    """_summary_

    Args:
        src (_type_): _description_
        dst (_type_): _description_
    """
    await src.seek(0)
    with open(dst, "wb") as buffer:
        while True:
            contents = await src.read(CHUNK_SIZE)
            if not contents:
                break
            buffer.write(contents)


def delete_file(filename):
    """_summary_

    Args:
        filename (_type_): _description_
    """
    os.remove(filename)


@app.get("/")
async def home():
    """_summary_

    Returns:
        _type_: _description_
    """

    return {"Message": "Hello World!"}


@app.post("/generateOuterImages")
async def generate_images(file: UploadFile):
    """_summary_

    Args:
        file (UploadFile): _description_

    Returns:
        _type_: _description_
    """
    pyplot.ion()
    figure = pyplot.figure(facecolor='black')
    axes = figure.add_subplot(projection='3d')
    axes.set_facecolor("black")
    figure.subplots_adjust(bottom=0., left=0.0, right=1., top=1.)
    path = os.getcwd()
    file_location = path + f"/{file.filename}'"
    await chunked_copy(file, file_location)
    your_mesh = mesh.Mesh.from_file(file_location)
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(
        your_mesh.vectors, color='white'))
    scale = your_mesh.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)
    axes.set_aspect('auto')
    axes.grid(False)
    axes._axis3don = False
    pyplot.tight_layout()
    pyplot.show()
    i = 1
    # images = []
    vx = [1, -1, 0, 0, 0, 0, 0.7071, -0.7071, -0.7071, 0.7071,
          0, 0, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0, 0]
    vy = [0, 0, 1, -1, 0, 0, 0.7071, -0.7071, 0.7071, -0.7071, 0.7071, -
          0.7071, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.7071, 0.07071]
    vz = [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0.7071, -0.7071, 0.7071, -0.7071,
          0.7071, -0.7071, -0.7071, 0.7071, -0.7071, 0.7071, 0.7071, -0.7071]
    for i in range(0, 22):
        x = vx[i]
        y = vy[i]
        elevation = 90
        if x != 0:
            elevation = math.degrees(math.atan(y/x))
        angle = math.degrees(math.acos(vz[i]))
        axes.view_init(elevation, angle)
        pyplot.draw()
        pyplot.pause(.001)
        figure.savefig(
            './images/'+f'ViewPoint{i}'+'.jpg')
        # images.append(FileResponse(os.path.abspath(os.path.join(
        #     os.path.dirname(__file__), f'images/ViewPoint{i}'))))
    pyplot.close("all")
    delete_file(file_location)

    return {"Message": "22 images has been generated"}


@app.post('/generateSlicedImages')
async def generate_sliced_images(file: UploadFile, axis: str, step: float):
    """Api endpoint to generate the sliced images of the model.

    Args:
        file (UploadFile): input .stl file

    Returns:
        string: Message on successful generation of sliced images.
    """
    print(axis, step)
    x = 0
    y = 0
    z = 0
    if axis == 'X':
        x = step
    elif axis == 'Y':
        y = step
    else:
        z = step
    path = os.getcwd()
    file_location = path + f"/{file.filename}"
    await chunked_copy(file, file_location)
    meshx = trimesh.load_mesh(file_location)
    z_extents = meshx.bounds[:, 2]
    # slice every .11811 model units (eg, inches)
    z_levels = np.arange(*z_extents, step=1)
    sections = meshx.section_multiplane(plane_origin=meshx.bounds[0],
                                        plane_normal=[x, y, z],
                                        heights=z_levels)
    sections = [ele for ele in sections if ele is not None]
    for idx, ele in enumerate(sections, 0):
        plt.axis('off')
        fig = plt.figure(1)
        fig.patch.set_facecolor('black')
        axes = fig.add_subplot(111)
        axes.set_facecolor('black')
        for patch in ele.polygons_full:
            axes.add_patch(PolygonPatch(patch, fc='w'))

        for patch in (ele.polygons_closed):
            plt.plot(*(patch.exterior.xy), 'k')
        plt.savefig('./slicedImages/' + f'meshx_slice{str(idx)}.png')
        plt.close()
    delete_file(file_location)
    return {
        "count": len(sections),
        "Message": f'{len(sections)} sliced images has been generated'
    }


@app.get("/getSlicedMatrix")
async def get_sliced_matrix(size: int):
    """Api enpoint to get the dissimilarity matrix and external shape
     complexity.
    """
    print(size)
    lst = []
    for i in range(0, size):
        for j in range(0, size):
            if i < 10:
                path1 = './slicedImages/' + f'meshx_slice{str(i)}.png'
            else:
                path1 = './slicedImages/' + f'meshx_slice{str(i)}.png'
            img = cv2.imread(path1, 0)
            ret, thresh1 = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
            contours1, hierarchy = cv2.findContours(
                thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt = 0.001
            if len(contours1) > 0:
                cnt = contours1[0]
            if j < 10:
                path = './slicedImages/' + f'meshx_slice{str(j)}.png'
            else:
                path = './slicedImages/' + f'meshx_slice{str(j)}.png'
            img2 = cv2.imread(path, 0)
            ret, thresh = cv2.threshold(img2, 128, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt1 = 0.001
            if len(contours) > 0:
                cnt1 = contours[0]
            ret = cv2.matchShapes(cnt, cnt1, cv2.CONTOURS_MATCH_I2, 0)
            lst.append("%.4f" % ret)
    matrix = []
    while len(lst) != 0:
        matrix.append(lst[:size])
        lst = lst[size:]
    mat = (np.asarray(matrix))
    print("slicedImages_LCM", mat.shape)
    workbook = xlsxwriter.Workbook('slicedImages_LCM.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0

    for col, data in enumerate(matrix):
        worksheet.write_column(row, col, data)

    workbook.close()
    return {
        "Message": "Sliced images dissimilirity matrix calculated successfully!"
    }


@app.get("/getOuterMatrix")
async def get_outer_matrix():
    """_summary_
    """
    start = time.time()
    lst = []
    SIZE = 22
    for i in range(0, SIZE):
        for j in range(0, SIZE):
            if i < 10:
                path1 = './images/ViewPoint' + str(i)+'.jpg'
            else:
                path1 = './images/ViewPoint' + str(i)+'.jpg'
            img = cv2.imread(path1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh1 = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
            Hu_1 = cv2.HuMoments(cv2.moments(thresh1)).flatten()
            for k in range(0, 7):
                Hu_1[k] = (-1 * math.copysign(1.0, Hu_1[k])
                           * math.log10(abs(Hu_1[k])))
            if j < 10:
                path = './images/ViewPoint' + str(j)+'.jpg'
            else:
                path = './images/ViewPoint' + str(j)+'.jpg'
            img2 = cv2.imread(path)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(img2, 128, 255, cv2.THRESH_BINARY)
            Hu_2 = cv2.HuMoments(cv2.moments(thresh)).flatten()
            for l in range(0, 7):
                Hu_2[l] = (-1 * math.copysign(1.0, Hu_2[l])
                           * math.log10(abs(Hu_2[l])))
            ret = np.sqrt(np.sum(np.square(Hu_2-Hu_1)))
            lst.append("%.4f" % ret)
    matrix = []
    while len(lst) != 0:
        matrix.append(lst[:SIZE])
        lst = lst[SIZE:]
    m = (np.asarray(matrix))
    workbook = xlsxwriter.Workbook('outerImages_ESCM.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0

    for col, data in enumerate(matrix):
        worksheet.write_column(row, col, data)

    workbook.close()

    end = time.time()

    elapsed = end - start
    print("Elapsed Time", elapsed)
    return {
        "Message": "External images dissimilirity matrix calculated successfully!"
    }


@app.get('/downloadOuterMatrix')
async def download_outer_matrix():
    """_summary_

    Returns:
        _type_: _description_
    """
    headers = {'Content-Disposition': 'attachment; filename="Book.xlsx"'}
    return FileResponse('./outerImages_ESCM.xlsx', headers=headers)


@app.get('/downloadSlicedMatrix')
async def download_sliced_matrix():
    """_summary_

    Returns:
        _type_: _description_
    """
    headers = {'Content-Disposition': 'attachment; filename="Book.xlsx"'}
    return FileResponse('./slicedImages_LCM.xlsx', headers=headers)


@app.get("/shapeComplexity")
async def get_shape_complexity():
    """_summary_

    Returns:
        _type_: _description_
    """
    comp1 = None
    comp2 = None
    comp1 = calc.calculate_shape_complexity("./outerImages_ESCM.xlsx")
    comp2 = calc.calculate_shape_complexity("./slicedImages_LCM.xlsx")
    comb = 0.5 * comp1 + 0.5 * comp2
    delete_file("./outerImages_ESCM.xlsx")
    delete_file("./slicedImages_LCM.xlsx")
    data = {"OuterImage": comp1, "SlicedImage": comp2, "combined": comb}
    return data

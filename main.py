"""
    This is the main file of stl_viewer backend module.
"""
import os,json
import math
import time
import trimesh
from fastapi import FastAPI, UploadFile,File,Form,Request
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
import networkx as nx
import pandas as pd
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import multipart
from typing import List
from io import BytesIO

from alcoa import alcoa

origins = [
    "http://127.0.0.1",
    "http://127.0.0.1:4001",
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
layer_thickness=0.03 #mm


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
async def get_shape_complexity(internalWeightage: float, externalWeightage: float):
    """_summary_

    Args:
        internalWeightage (float): _description_
        externalWeightage (float): _description_

    Returns:
        _type_: _description_
    """
    comp1 = None
    comp2 = None
    comp1 = calc.calculate_shape_complexity("./outerImages_ESCM.xlsx")
    comp2 = calc.calculate_shape_complexity("./slicedImages_LCM.xlsx")
    comb = comp1 * internalWeightage + comp2 * externalWeightage
    delete_file("./outerImages_ESCM.xlsx")
    delete_file("./slicedImages_LCM.xlsx")
    message = "Selected for redesign"
    if comp2 >= 1 or comb >= 4:
        message = "Design can be selected for AM"
    data = {"OuterImage": round(comp1, 4), "SlicedImage": round(
        comp2, 4), "combined": round(comb, 4), "Message": message}
    return data


@app.post("/getWeights")

async def getWeights(file:UploadFile=File(...)):

    print(file)
    contents = await file.read()
    print(contents)
    # print(file.filename)
    # df = pd.read_csv('steering.csv')
    df = pd.read_csv(BytesIO(contents))
    print(df)
    Graphtype = nx.Graph()
    G = nx.from_pandas_edgelist(df, edge_attr='weight', create_using=Graphtype)
    nx.draw(G, with_labels=True)

    node = nx.degree_centrality(G)
    return JSONResponse(content=jsonable_encoder(node))

# cost_mat,material_density,support_vol,num_layers
def support_cost_calc(cost_mat,material_density,support_vol,num_layers):
    des_time=1 # hr
    wage_des_engr=27 # $/hr
    avg_incidental_wage=33.91 # $/hr

    equ_usage_time=1 # hr
    sw_license_cost=8822 # $/annum
    des_sw_util= 2628 # hrs/annum
    hw_investment_cost=980 # $
    hw_util=6570 # hrs/annum
    hw_amor_period= 3 #yrs

    office_area=3 #m^2
    office_util= 5256 # hrs/annum
    avg_rent_office=14.6 # $/m^2
    avg_op_cost_office=3.05 # $/m^2

    cost_stress=350 # $
    cost_EDM=200 # $
    time_post_proc=2 # hrs
    cost_operator=110 # $/hr
    cost_tools=50 # $
    # material_density=4410 # modifiable param
    # cost_mat=680 # modifiable param
    cost_scrap=5 # $/kg
    # support_vol=1 # modifiable param
    # num_layer=1 # modifiable param
    hatch_spacing=0.15 #mm
    
    offset_height_support=2 #mm
    recoating_time=10 #sec
    delay_time=2 #sec
    cost_gas_filter_machine_invest=328 # $
    machine_util=4380 #hrs/annum
    machine_filter_amor_period=5 #yrs
    cost_gas=4.9 #$
    cost_energy=1.54 #$
    machine_cost=490100 #$
    machine_amor_period=5 #yrs
    machine_area=12.6 #m^2
    avg_machine_space_rent=14.6 #$/m^2
    avg_machine_space_op_cost=3.05 #$/m^2
    machine_service_cost=29406 #$/annum

    cost_op_des=des_time*(wage_des_engr+avg_incidental_wage)
    # print('cost-op-des',cost_op_des)
    cost_equipment_des= (equ_usage_time*sw_license_cost/des_sw_util)+((equ_usage_time*hw_investment_cost)/(hw_util*hw_amor_period))
    # print('cost equ des',cost_equipment_des)
    cost_office_space_des=(equ_usage_time*office_area/office_util)*(avg_rent_office+avg_op_cost_office)
    cost_design_support=cost_op_des+cost_equipment_des+cost_office_space_des

    print('material density',material_density)
    cost_material=material_density*support_vol*(cost_mat-cost_scrap) # whats support_vol??
    # print('cost office',cost_design_support)
    build_speed=500 #mm/sec
    melt_rate=build_speed*hatch_spacing*layer_thickness*3.6# whats nl??
    build_time=(support_vol/melt_rate)+((offset_height_support/layer_thickness)*(recoating_time+delay_time)/3600)
    cost_prod=(build_time/machine_util)*((machine_cost/machine_amor_period)+(machine_area*(avg_machine_space_rent+avg_machine_space_op_cost))+machine_service_cost)
    cost_gas_filter=(build_time*cost_gas_filter_machine_invest)/(machine_util*machine_filter_amor_period)
    cost_post_proc_gas=build_time*cost_gas
    cost_energy_total=build_time*cost_energy
    cost_manu_support=cost_material+cost_prod+cost_gas_filter+cost_post_proc_gas+cost_energy_total

    cost_substrate=cost_stress+cost_EDM
    cost_post_proc=time_post_proc*(cost_operator+cost_tools)
    cost_post_proc_support=cost_substrate+cost_post_proc
    print('ehllo',cost_prod)

    cost_proc_support=cost_design_support+cost_manu_support+cost_post_proc_support

    return round(cost_proc_support,2)


# part_vol,support_vol,num_layers
def proc_cost_calc(part_vol,support_vol,num_layers,material,material_cost):
    print(material)
    wrought_density={'ti64':4.41,'ss':7.8,'in625':5}
    tap_density={'ti64':2.74,'ss':5.3,'in625':5}
    build_rate=13.5/60 # min/cm^3
    time_setup=2 #hr
    recoat_rate=9 #sec/layer
    cost_am_machine=60 #$/hr
    cost_gas=10 #$/hr
    operator_rate=110 #$/hr
    unit_cost_mat=material_cost #$/kg
    time_removal=3 #hr
    powder_wrought_density=wrought_density[material] #g/cm^3
    powder_tap_density=tap_density[material] #g/cm^3
    # where to get these values from
    # part_vol=1 
    # support_vol=1
    # num_layers=1
    print('unit cost mat',unit_cost_mat)
    time_recoat=recoat_rate*num_layers/3600
    build_time=build_rate*(part_vol+support_vol)+time_recoat
    print('part vol',part_vol)
    print('supp vol',support_vol)
    print('build time',build_time)
    cost_operation=build_time*(cost_am_machine+cost_gas)
    cost_setup=time_setup*(cost_am_machine+operator_rate)
    mass=(1.4*powder_wrought_density*(part_vol+support_vol))+(0.25*powder_tap_density*support_vol)
    cost_mat=mass*unit_cost_mat
    cost_removal=time_removal*(cost_am_machine+operator_rate) 
    cost_proc=cost_mat+cost_setup+cost_operation+cost_removal
    print(cost_mat,cost_removal,cost_setup,cost_operation)
    return round(cost_proc,2)

@app.post('/costCalc')
async def costCalc(request: Request):   
    json_data = await request.body()
    a=BytesIO(json_data)
    data=json.loads(a.getvalue())
    
    material_id=['ti64','ss','in625']
    cost_proc=proc_cost_calc(part_vol=data['part_volume'],support_vol=data['support_volume'],num_layers=data['num_layers'],material=material_id[data['material']-1],material_cost=data['material_cost']/1000)
    cost_support=support_cost_calc(support_vol=data['support_volume'],num_layers=data['num_layers'],cost_mat=data['material_cost']/1000,material_density=data['material_density'])
    return {"cost_proc":round(cost_proc,2),"cost_support":round(cost_support,2)}



# External Images Complexity Metric: 
# Sliced Images Complexity Metric: 
@app.post('/multFile')
async def multFiles(single_file: UploadFile = File(...), multiple_files: List[UploadFile] = File(...),material:str=Form(...),material_cost:str=Form(...),material_density:str=Form(...)):
    # json_data = await request.body()
    # a=BytesIO(json_data)
    # data=json.loads(a.getvalue())
    print(material)
    # Read the contents of the single STL file
    single_file_contents = await single_file.read()

#     # Read the contents of each of the multiple STL files
    multiple_file_contents = []
    for file in multiple_files:
        file_contents = await file.read()
        multiple_file_contents.append(file_contents)
    material_id=['ti64','ss','in625']
    # print(single_file_contents)
    file_location = f"uploads/{single_file.filename}"
    with open(file_location, "wb") as file_object:
        file_object.write(single_file_contents)
    model_height=getHeight(file_location)
    g=getPartVol(file_location)
    print(g)
    single={}

    single["cost_proc"]=proc_cost_calc(part_vol=g,support_vol=1,num_layers=model_height/layer_thickness,material_cost=float(material_cost)/1000,material=material_id[int(material)])
    single["cost_support"]=support_cost_calc(cost_mat=float(material_cost)/1000,material_density=float(material_density),support_vol=1,num_layers=model_height/layer_thickness)

    print(len(multiple_file_contents))
    final_data=[]
    for i in range(0,len(multiple_file_contents),5):
        print(f"WE in {i}")
        file_location = f"uploads/{multiple_files[i].filename}"
        with open(file_location, "wb") as file_object:
            file_object.write(multiple_file_contents[i])
        model_height=getHeight(file_location)
        g=getPartVol(file_location)
        print(g)
        a= await generate_images(multiple_files[i])
        print(a)
        b= await generate_sliced_images(multiple_files[i],"X",1)
        print(b)
        c=await get_outer_matrix()
        print(c)
        d=await get_sliced_matrix(b['count'])
        print(d)
        e=await get_shape_complexity(internalWeightage=0.5,externalWeightage=0.5)
        print(e)
        e["cost_proc"]=proc_cost_calc(part_vol=g,support_vol=1,num_layers=model_height/layer_thickness,material_cost=float(material_cost)/1000,material=material_id[int(material)])
        e["cost_support"]=support_cost_calc(cost_mat=float(material_cost)/1000,material_density=float(material_density),support_vol=1,num_layers=model_height/layer_thickness)
        final_data.append(e)

    return {'message':'success',"data":{"multiple":final_data,"single":single},"status":200}

def find_mins_maxs(obj):
    minx = obj.x.min()
    maxx = obj.x.max()
    miny = obj.y.min()
    maxy = obj.y.max()
    minz = obj.z.min()
    maxz = obj.z.max()
    return minx, maxx, miny, maxy, minz, maxz

def getHeight(filepath):
    # print(filepath)
    main_body = mesh.Mesh.from_file(filepath)
    print(find_mins_maxs(main_body))
    minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(main_body)
    w1 = maxx - minx
    l1 = maxy - miny
    h1 = maxz - minz
    return h1

def getPartVol(filepath):
    my_mesh = trimesh.load(filepath)
    vol = my_mesh.volume

    cmvol = vol/1000
    return cmvol

@app.post('/rankJava')
async def rankJavaCode(data:dict):
    # json_data = await request.body()
    # # print(json_data)
    # a=BytesIO(json_data)
    # print(a)
    # data=json.load(a)
    # print(data)
    escm = data.get('escm', [])
    lcm = data.get('lcm', [])
    cbr = data.get('cbr', [])
    ic = data.get('ic', [])
    print(escm)
    print(lcm)
    print(cbr)
    print(ic)
    alcoa(escm,lcm,cbr,ic)
    return {"message":"success","status":200}
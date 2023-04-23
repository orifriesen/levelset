import os
from osgeo import ogr, osr, gdal
import numpy as np
import matplotlib.pyplot as plt
import invr
import time
import logging

def rasterize_shp(city, race, max_pixels):
    script_dir = os.path.dirname(__file__)
    rel_path = 'data/shapefiles/' + city + '_' + race + '.shp'

    shp = os.path.join(script_dir, rel_path)

    source_ds = gdal.OpenEx(shp, gdal.OF_VECTOR | gdal.OF_UPDATE)
    source_ds.ExecuteSQL("ALTER TABLE " + city + '_' + race + " DROP COLUMN isMaj")

    source_ds = None

    source_ds = ogr.Open(shp, 1)
    source_layer = source_ds.GetLayer()

    fd = ogr.FieldDefn('isMaj', ogr.OFTInteger)
    source_layer.CreateField(fd)

    for feat in source_layer:
        is_maj = (feat.GetField('percent') > 50)
        feat.SetField('isMaj', (1 - int(is_maj)) * 255)
        source_layer.SetFeature(feat)

    x_min, x_max, y_min, y_max = source_layer.GetExtent()
    max_cols = max_rows = max_pixels
    max_pixel_width = (x_max - x_min) / max_cols
    max_pixel_height = (y_max - y_min) / max_rows
    pixel_width = pixel_height = max(max_pixel_width, max_pixel_height)
    cols = int((x_max - x_min) / pixel_height)
    rows = int((y_max - y_min) / pixel_width)

    rel_path = 'data/tif/' + race + '/' + city + '/' + city + '.tif'
    out_tiff = os.path.join(script_dir, rel_path)

    target_ds = gdal.GetDriverByName('Gtiff').Create(out_tiff, cols, rows, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((x_min, pixel_width, 0, y_max, 0, -pixel_height))
    band = target_ds.GetRasterBand(1)
    no_data_value = 255
    band.SetNoDataValue(no_data_value)
    band.FlushCache()

    gdal.RasterizeLayer(target_ds, [1], source_layer, options=["ATTRIBUTE=isMaj"])
    target_ds_srs = osr.SpatialReference()
    target_ds_srs.ImportFromEPSG(4326)
    target_ds.SetProjection(target_ds_srs.ExportToWkt())

    source_ds = None

    source_ds = gdal.OpenEx(shp, gdal.OF_VECTOR | gdal.OF_UPDATE)
    source_ds.ExecuteSQL("ALTER TABLE " + city + '_' + race + " DROP COLUMN isMaj")

    source_ds = None


def godunov(phi, a):
    phi_p_x = np.zeros(phi.shape)
    phi_n_x = np.zeros(phi.shape)
    phi_p_y = np.zeros(phi.shape)
    phi_n_y = np.zeros(phi.shape)

    phi_p_x[:, 0:-1] = phi[:, 1:] - phi[:, 0:-1]
    phi_n_x[:, 1:] = phi[:, 1:] - phi[:, 0:-1]
    phi_p_y[0:-1, :] = phi[1:, :] - phi[0:-1, :]
    phi_n_y[1:, :] = phi[1:, :] - phi[0:-1, :]

    phi_x_neg = np.maximum(np.power(np.maximum(phi_n_x, 0), 2), np.power(np.minimum(phi_p_x, 0), 2))
    phi_y_neg = np.maximum(np.power(np.maximum(phi_n_y, 0), 2), np.power(np.minimum(phi_p_y, 0), 2))

    phi_x_pos = np.maximum(np.power(np.minimum(phi_n_x, 0), 2), np.power(np.maximum(phi_p_x, 0), 2))
    phi_y_pos = np.maximum(np.power(np.minimum(phi_n_y, 0), 2), np.power(np.maximum(phi_p_y, 0), 2))

    phi_x = -np.minimum(np.sign(a), 0) * phi_x_neg + np.maximum(np.sign(a), 0) * phi_x_pos
    phi_y = -np.minimum(np.sign(a), 0) * phi_y_neg + np.maximum(np.sign(a), 0) * phi_y_pos
    return np.sqrt(phi_x + phi_y)


def sign(phi):
    denom = np.sqrt(np.power(phi, 2) + godunov(phi, np.ones(phi.shape)))
    return np.divide(phi, denom)


def normalize_distance(image, time):
    length, width = image.shape
    c = np.sqrt(length ^ 2 + width ^ 2)
    distances = image.copy()
    distances = distances.astype(float)
    distances = ((distances / 255) * 2 - 1) * c
    sign_phi = sign(distances)
    dt = 1 / (4 * np.max(np.abs(sign_phi)))

    for i in np.arange(time / dt):
        dphi = dt * sign_phi * (1 - godunov(distances, -sign_phi))
        distances = distances + dphi

    return distances


def advance_level_set(image, time):
    a = np.ones(image.shape)
    dt = .125

    for i in np.arange(time / dt):
        dphi = -dt * a * godunov(image, a)
        image = image + dphi

    return image


def write_img_idx_map(image, existing_verts, entry_times, T):
    curr_verts = np.flatnonzero(image==0)
    r,c = image.shape
    curr_verts = [vert for vert in curr_verts if ((vert%c)%5==0 and (int(vert/c))%5==0)]
    new_verts = list(set(curr_verts).difference(set(existing_verts.keys())))
    n = len(existing_verts)
    for vert in new_verts:
        existing_verts[vert] = n
        n += 1
        entry_times.append(T)
    return existing_verts, entry_times


def gen_img_adjacencies(existing_verts, h, w):
    existing_verts_list = list(existing_verts.keys())
    adjacencies = [[]]*len(existing_verts)
    for vert in existing_verts_list:
        poss_adj = [vert-5*w-5, vert-5*w, vert-5, vert+5, vert+5*w, vert+5*w+5]
        poss_adj = [i for i in poss_adj if i>=0]
        poss_adj = list(set(poss_adj).intersection(set(existing_verts_list)))
        adjacencies[existing_verts[vert]] = [existing_verts[neighbor] for neighbor in poss_adj]
    return adjacencies

def build_levelset_complex(city, race, is_tiff = True, save_plots = False):
    if not is_tiff:
        rasterize_shp(city, race, 250)

    rel_path = 'data/tif/' + race + '/' + city + '/' + city + '.tif'
    img_array = plt.imread(os.path.join(os.path.dirname(__file__), rel_path))
    # img_array = np.zeros((6,11))
    # img_array[5,5] = 255

    dT = .5
    n = 40
    T = 0
    existing_verts = {}
    entry_times = []
    existing_verts, entry_times = write_img_idx_map(img_array, existing_verts, entry_times, T)

    gamma = 7
    img_array = normalize_distance(img_array, gamma)
    for i in range(n):
        T = T + dT
        img_array = advance_level_set(img_array, dT)
        phi_array_img = (np.sign(img_array) + 1) / 2 * 255


        if save_plots:
            rel_path = 'data/tif/' + race + '/' + city + '/' + city + '_' + str(T) + '.tiff'
            T_out = os.path.join(os.path.dirname(__file__), rel_path)
            phi_array_img = phi_array_img.astype(int)
            plt.imsave(T_out, phi_array_img)
        if i % 5 == 0:
            img_array = normalize_distance(phi_array_img, gamma)
        existing_verts, entry_times = write_img_idx_map(phi_array_img, existing_verts, entry_times, T)
    phi_array_img = np.zeros(img_array.shape)
    phi_array_img = phi_array_img.astype(int)
    existing_verts, entry_times = write_img_idx_map(phi_array_img, existing_verts, entry_times, 10.5)

    key_write_filename = 'data/keys/ls/' + race + '/' + city + '-key'
    with open(os.path.join(os.path.dirname(__file__), key_write_filename), 'w') as key_file:
        for k, v in existing_verts.items():
            key_file.write(str(k)+','+str(v)+'\n')
    key_file.close()

    h, w = img_array.shape
    adjacencies = gen_img_adjacencies(existing_verts, h, w)

    maxDimension = 3
    V = []
    V = invr.incremental_vr(V, adjacencies, maxDimension)
    outputDirPath = 'data/sc/ls/' + race + '/'
    outputDir = os.path.join(os.path.dirname(__file__), outputDirPath)
    entry_times = [int(time*2) for time in entry_times]

    entryTimesSub = [entry_times[max(simplex)-1] for simplex in V]

    np.savetxt(outputDir + city + '_entry_times.csv',entryTimesSub,
               delimiter=' ',fmt='%i')

    phatFormatV = invr.replace_face(V)

    F = open(outputDir + city + ".dat", "w")

    for face in phatFormatV:
        F.write(str(len(face) - 1))
        if len(face) > 1:
            for simplex in face:
                F.write(" " + str(simplex))
        F.write("\n")
    F.close()

def main():
    #logging.basicConfig(filename='../logs/ls.log', filemode='w+', level=logging.WARNING)
    #timing_csv = '../runtimes/ls_times.csv'
    #with open(timing_csv,'w+') as timing_file:
    #with open('../full-list') as county_file:
    cities = ['okc', 'seattle', 'stpaul', 'pittsburgh', 'cleveland', 'indianapolis', 'chicago', 'cincinnati', 'philadelphia', 'rochester', 'hartford', 'boston', 'sacramento', 'lasvegas', 'denver', 'portland', 'jackson', 'batonrouge', 'birmingham', 'tulsa']
    for city in cities:
    # for county in ['003-alpine']:
        for race in ['white','black', 'asian', 'hispanic']:
            #start_time = time.time()
            build_levelset_complex(city, race, False, True)
            #timing_file.write(county + ',ls,'+candidate+','+str(time.time()-start_time)+'\n')


if __name__ == '__main__':
    main()

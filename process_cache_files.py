import json
import os
import copy
import statistics
import numpy as np


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = "/".join(current_dir.split('/')[:-1]) + "/"

    # Read file
    # We do hyperparameter tuning on GTX 1080Ti files
    data_path = root_dir + 'GPU_benchmarking_paper/cache_files/'
    processed_data_path = root_dir + 'GPU_benchmarking_paper/processed_cache_files/'

    FJ_files = ['convolution_A100_FJ.json']

    convolution_files = ['MI50_convolution_15x15.json',
    'convolution_GTX_1080Ti.json',
    'convolution_A100.json',
    'convolution_RTX_2070_SUPER.json',
    'convolution_TITAN_RTX.json',
    'convolution_V100.json',
    'convolution_P100.json',
    'convolution_K20.json',
    'convolution_GTX_Titan_X.json']

    GEMM_files = ['GEMM_A100.json',
    'GEMM_GTX_1080Ti.json',
    'GEMM_RTX_2070_SUPER.json',
    'GEMM_TITAN_RTX.json',
    'GEMM_V100.json',
    'MI50_GEMM_cache.json',
    'GEMM_P100.json',
    'GEMM_K20.json',
    'GEMM_GTX_Titan_X.json']

    pnpoly_files = ['pnpoly_A100.json',
    'pnpoly_GTX_1080Ti.json',
    'pnpoly_RTX_2070_SUPER.json',
    'pnpoly_TITAN_RTX.json',
    'pnpoly_V100.json',
    'pnpoly_P100.json',
    'pnpoly_K20.json',
    'pnpoly_GTX_Titan_X.json',
    ]

    for filename in convolution_files[2:3]:
        with open(data_path + filename, 'r') as myfile:
            data=myfile.read()
        data = json.loads(data)
        print(data['tune_params'])

        average_stdev = 0.0
        N = 0
        compiled_points = 0
        for key, val in data['cache'].items():
            meantime = val['time']
            if meantime < 1e10:
                compiled_points += 1
            if 'times' in val.keys():
                normalized_times = (np.array(val['times'])/float(meantime))
                stdev = statistics.stdev(normalized_times)
                average_stdev += stdev
                N += 1
        print('Average normalized stdev of runtime:', average_stdev/float(N))
        print("Number of valid points in space:", compiled_points)
        continue

        print("Device: " + str(data['device_name']))
        print("Kernel name: " + str(data['kernel_name']))
        print("Tunable parameters: " + str(data['tune_params_keys']), end='\n\n')

        # Pre-process the search space
        searchspace = data['tune_params']
        print("There are", len(data['cache'].keys()), "keys in the searchspace")

        for k in data['cache'].keys():
            try:#Power is recorderd if it is a valid kernel setting
                data['cache'][k].pop('power')
            except:
                continue
            try:
                data['cache'][k].pop('energy')
            except:
                continue

        # If want to do WHITEBOX
        restrict_space = False
        if restrict_space:
            new_dict = copy.deepcopy(data)
            # The restrictions are
            # ['block_size_x*block_size_y>=64', 'tile_size_x*tile_size_y<30']
            temp = []
            temp2 = []
            for k in data['cache'].keys():
                bs_x = data['cache'][k]['block_size_x']
                bs_y = data['cache'][k]['block_size_y']
                ts_x = data['cache'][k]['tile_size_x']
                ts_y = data['cache'][k]['tile_size_y']
                #print(bs_x, bs_y, ts_x, ts_y)
                if bs_x*bs_y not in temp:
                    temp.append(bs_x*bs_y)
                if ts_x*ts_y not in temp2:
                    temp2.append(ts_x*ts_y)
                if not bs_x*bs_y >= 64:
                    del new_dict['cache'][k]
                    raise Exception("PAUSE")
                if not ts_x*ts_y < 30:
                    del new_dict['cache'][k]
                    raise Exception("PAUSE")
                if not bs_x*bs_y <= 1024:
                    del new_dict['cache'][k]

            temp.sort()
            temp2.sort()
            print(temp)
            print(temp2)
            print(len(data['cache'].keys()))
            print(len(new_dict['cache'].keys()))
            data = new_dict

        if not restrict_space:
            newfilename = filename[:-5] + '_processed' + '.json'
        else:
            newfilename = filename[:-5] + '_processed_whitebox' + '.json'
        with open(processed_data_path + newfilename, 'w') as outfile:
            json.dump(data, outfile)

import json
import os

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = "/".join(current_dir.split('/')[:-1]) + "/"

    # Read file
    # We do hyperparameter tuning on GTX 1080Ti files
    data_path = root_dir + 'GPU_benchmarking_paper/cache_files/'
    processed_data_path = root_dir + 'GPU_benchmarking_paper/processed_cache_files/'

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

    for filename in convolution_files:
        with open(data_path + filename, 'r') as myfile:
            data=myfile.read()
        data = json.loads(data)

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

        newfilename = filename[:-5] + '_processed' + '.json'
        with open(processed_data_path + newfilename, 'w') as outfile:
            json.dump(data, outfile)

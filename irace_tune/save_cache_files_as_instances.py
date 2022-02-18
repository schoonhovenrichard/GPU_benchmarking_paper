import json
import os

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = "/".join(current_dir.split('/')[:-2]) + "/"

    data_path = root_dir + 'GPU_benchmarking_paper/processed_cache_files/'
    save_path = current_dir + '/stochastic_cache_files/'

    filename = 'convolution_P100_processed.json'
    with open(data_path + filename, 'r') as myfile:
        data=myfile.read()
    data = json.loads(data)

    keys = list(data.keys())
    cache_of_one_run = dict()
    for key in keys:
        if key == 'cache':
            continue
        cache_of_one_run[key] = data[key]

    first_entry = next(iter(data['cache'].items()))[1]
    timings = first_entry['times']
    base_filename = filename.split('.')[0]
    for run_idx in range(len(timings)):
        run_dict = cache_of_one_run.copy()
        run_dict['cache'] = {}
        for config, vals in data['cache'].items():
            if 'times' in vals.keys():
                run_time = vals['times'][run_idx]
                run_dict['cache'][config] = {}
                run_dict['cache'][config]['time'] = run_time

        save_filename = base_filename + '_run' + str(run_idx + 1) + '.json'
        with open(save_path + save_filename, 'w') as fp:
            json.dump(run_dict, fp)

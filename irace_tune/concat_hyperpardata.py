import csv
import os


if __name__ == '__main__':
    ### Get the files
    kernels = ['pnpoly', 'GEMM','convolution']
    GPUs = ['P100', 'GTX_1080Ti', 'RTX_2070_SUPER']

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = "/".join(current_dir.split('/')[:-1]) + "/"
    file_dir = root_dir + '/irace_tune/tune_files/'
    hyperpar_files = [f for f in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, f))]

    ## Order files per kernel and GPU model
    file_dict = {}
    for kernel in kernels:
        file_dict[kernel] = {}
        for gpu in GPUs:
            file_dict[kernel][gpu] = []
    for file in hyperpar_files:
        file_ls = file.split("_")
        kernel = file_ls[3]
        GPU = "_".join(file_ls[4:-4])
        file_dict[kernel][GPU].append(file)

    ## For every kernel/GPU combi, concat all lines of data in one CSV file
    for kernel in kernels:
        for gpu in GPUs:
            export_filename = 'tune_hyperpars_iRace_' + kernel + '_' + gpu + '_processed_runs=10.csv'
            file_count = 0
            export_csv = []
            for file in file_dict[kernel][gpu]:
                with open(file_dir + file) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    line_count = 0
                    if file_count == 0:
                        for row in csv_reader:
                            line_count += 1
                            export_csv.append(row)
                    else:
                        for row in csv_reader:
                            if line_count < 2:
                                line_count += 1
                            else:
                                export_csv.append(row)
                                line_count += 1
                    print(f'Processed {line_count} lines.')
                file_count += 1

            # Save the concatenated data in one CSV file
            with open(export_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(export_csv)

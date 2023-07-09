import os
import argparse
import glob
import fnmatch


def filter_files(directory, exclude_git=False, filters=None, extension=None, recursive=False, ignore=None):
    filtered_files = []

    if recursive:
        for root, _, _ in os.walk(directory):
            matching_files = glob.glob(os.path.join(root, '*'))

            for file_path in matching_files:
                if exclude_git and '.git' in file_path:
                    continue

                ignore_file = False
                if ignore:
                    for ignore_pattern in ignore:
                        if fnmatch.fnmatch(file_path, os.path.join(directory, ignore_pattern)):
                            ignore_file = True
                            break

                if ignore_file:
                    continue

                if extension is None or file_path.endswith(extension):
                    filtered_files.append(file_path)
    else:
        for filter_pattern in filters:
            filter_pattern = os.path.join(directory, filter_pattern)
            matching_files = glob.glob(filter_pattern)

            for file_path in matching_files:
                if exclude_git and '.git' in file_path:
                    continue

                ignore_file = False
                if ignore:
                    for ignore_pattern in ignore:
                        if fnmatch.fnmatch(file_path, os.path.join(directory, ignore_pattern)):
                            ignore_file = True
                            break

                if ignore_file:
                    continue

                if extension is None or file_path.endswith(extension):
                    filtered_files.append(file_path)

    return filtered_files


def write_files_content(file_paths):
    with open("files_content.txt", 'w') as file:
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                contents = f.read()
                file.write(f"-- FILE: {file_path} BEGINS\n")
                file.write(contents + '\n')
                file.write('-' * 100 + '\n')


# Create the argument parser
parser = argparse.ArgumentParser(
    description='Print files and their contents in a directory and subdirectories.')

# Add the 'start_directory' argument
parser.add_argument('--start_directory', metavar='start_directory',
                    type=str, help='the directory to start the search')

# Add the '--exclude-git' flag
parser.add_argument('--exclude-git', dest='exclude_git',
                    action='store_true', help='exclude the .git directory')

# Add the '--filter' flag with multiple values
parser.add_argument('--filter', dest='filter', type=str, nargs='+',
                    help='filter the files and directories to be printed')

# Add the '--ignore' flag with multiple values
parser.add_argument('--ignore', dest='ignore', type=str, nargs='+',
                    help='ignore the specified files or directories')

# Add the '--extension' argument
parser.add_argument('--extension', dest='extension', type=str,
                    help='filter files by extension')

# Add the '--recursive' flag
parser.add_argument('--recursive', dest='recursive',
                    action='store_true', help='search for files recursively')

# Add the '--list_files' flag
parser.add_argument('--list_files', dest='list_files',
                    action='store_true', help='list the files without printing their contents')

# Add the '--write' flag
parser.add_argument('--write', dest='write',
                    action='store_true', help='write the output to files_content.txt')

# Parse the command-line arguments
args = parser.parse_args()

# Call the function with the specified directory, flags, and extension
filtered_files = filter_files(args.start_directory,
                              exclude_git=args.exclude_git,
                              filters=args.filter,
                              extension=args.extension,
                              recursive=args.recursive,
                              ignore=args.ignore)

if args.write:
    write_files_content(filtered_files)

if args.list_files:
    for file_path in filtered_files:
        print(file_path)

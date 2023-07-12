import os
import argparse
import glob
import fnmatch


def should_exclude_path(path, exclude_git, ignore):
    if exclude_git and '.git' in path:
        return True

    if ignore:
        for ignore_pattern in ignore:
            if fnmatch.fnmatch(path, ignore_pattern):
                return True

    return False


def is_file_filtered(file_path, filters, extension):
    if filters:
        for filter_pattern in filters:
            if fnmatch.fnmatch(file_path, filter_pattern):
                return True

    if extension and not file_path.endswith(extension):
        return True

    return False


def get_files_in_directory(directory):
    return glob.glob(directory)


def get_filtered_files(directory, exclude_git=False, filters=None, extension=None, ignore=None):
    filtered_files = []

    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(root, filename)

            if os.path.isdir(file_path):
                continue

            if exclude_git and '.git' in file_path:
                continue

            if ignore:
                if any(fnmatch.fnmatch(file_path, pattern) for pattern in ignore):
                    continue

            if filters:
                if any(fnmatch.fnmatch(file_path, pattern) for pattern in filters):
                    if extension and not file_path.endswith(extension):
                        continue
                else:
                    continue
            elif extension and not file_path.endswith(extension):
                continue

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
filtered_files = get_filtered_files(args.start_directory,
                                    exclude_git=args.exclude_git,
                                    filters=args.filter,
                                    extension=args.extension,
                                    ignore=args.ignore)

if args.write:
    write_files_content(filtered_files)

if args.list_files:
    for file_path in filtered_files:
        print(file_path)

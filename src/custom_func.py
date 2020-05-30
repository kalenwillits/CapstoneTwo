
# Define read_large_file()
def read_large_file(file_object):
    """A generator function to read a large file lazily."""
    # Loop indefinitely until the end of the file
    while True:
        # Read a line from the file: data
        data = file_object.readline()
        # Break if this is the end of the file
        if not data:
            break
        # Yield the line of data
        yield data

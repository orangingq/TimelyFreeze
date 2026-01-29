import os
def get_abs_path(path:str, base_dir:str)->str:
    '''Get the absolute path of the specified path. 
    Args:
        path (str): the path to be joined with the default path.
        base_dir (str): base directory name to join if path is relative. (E.g. "/home/user/dump_folder" -> "/home/user/dump_folder/{path}")
    Returns:
        str: the absolute path.
    '''
    if not os.path.isabs(path):
        path = os.path.abspath(os.path.join(base_dir, path))
    else:
        path = os.path.normpath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

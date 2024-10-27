from pathlib import Path
try:
    from tu.loggers.utils import print_vcv_url
except:
    print_vcv_url = lambda *args, **kwargs: print('[INFO]', str(args) + str(kwargs))

import time
import uuid
import re
import hashlib
from unidecode import unidecode


def setup_save_dir(save_dir: str, log_unique: bool) -> Path:
    save_dir = Path(save_dir)
    if log_unique:
        timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
        save_dir = save_dir.with_stem(f'{Path(save_dir).stem}_{timestamp}_{uuid.uuid4()}')
        save_dir.mkdir(parents=True)
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
    print_vcv_url(save_dir.as_posix())
    return save_dir


def modify_string_for_file(orig_string, max_length=100, append_uuid=True):
    """
    Truncates the string to a maximum length, appends a UUID derived from the string,
    and sanitizes it to be safe for use as a file or folder name.

    Args:
    orig_string (str): The original string to modify.
    max_length (int): The maximum length of the string before truncation.

    Returns:
    str: A modified string safe for use as a file or folder name.
    """
    # Generate a UUID based on the original string
    string_uuid = '' if not append_uuid else str(uuid.uuid5(uuid.NAMESPACE_DNS, orig_string))

    # Sanitize the string to be safe for file/folder names, removing problematic characters
    ascii_string = unidecode(orig_string).replace(" ", "_")
    sanitized_string = re.sub(r'[<>:"/\\|?*.\']', "", ascii_string).strip()

    # If the sanitized string is empty, use "default" to ensure a valid file name
    if not sanitized_string:
        sanitized_string = "default"

    # Prepare the final string, ensuring it does not exceed the maximum length
    result_string = f"{sanitized_string}_{string_uuid}"
    if len(result_string) > max_length:
        # Cut the sanitized part to fit the UUID and ensure total length compliance
        cut_length = max_length - len(string_uuid) - 1  # Subtracting 1 for the underscore
        result_string = f"{sanitized_string[:cut_length]}_{string_uuid}"

    return result_string


def path_to_unique_string(file_path: Path):
    # Encode the file path as bytes
    file_path = file_path.resolve().as_posix()
    path_bytes = file_path.encode('utf-8')

    # Create a SHA-256 hash object
    hash_object = hashlib.sha256(path_bytes)

    # Get the hexadecimal representation of the hash
    unique_string = hash_object.hexdigest()

    return unique_string

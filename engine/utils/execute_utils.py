import subprocess
import time
import sys
from pathlib import Path


def execute_command(command: str, save_dir: str, timeout=None, dry_run: bool = False,
                    print_stdout: bool = False, print_stderr: bool = False, cwd=None) -> int:
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    with open((save_dir / 'impl.sh').as_posix(), 'w') as f:
        f.write(command)
    print(f"[INFO] Executing command: \n{command}")
    print(f'[INFO] Outputs will be saved to {save_dir.resolve().as_posix()}')
    if dry_run:
        print("[INFO] Dry run, skipping execution.")
        return 0

    # with open(save_dir / "execute_out.txt", 'w') as out_file, open(save_dir / "execute_err.txt", 'w') as err_file:
    #     process = subprocess.Popen(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    #
    #     try:
    #         while True:
    #             output = process.stdout.readline()
    #             error = process.stderr.readline()
    #
    #             if output == '' and error == '' and process.poll() is not None:
    #                 break
    #
    #             if output:
    #                 if print_stdout:
    #                     print(output, end='')
    #                 out_file.write(output)
    #                 out_file.flush()
    #
    #             if error:
    #                 if print_stderr:
    #                     print(error, end='', file=sys.stderr)
    #                 err_file.write(error)
    #                 err_file.flush()
    #             time.sleep(10)
    #
    #         process.wait(timeout=timeout)
    #         returncode = process.returncode
    #
    #     except subprocess.TimeoutExpired:
    #         process.kill()
    #         out_file.write("\nProcess timed out")
    #         err_file.write("\nProcess timed out")
    #         returncode = -1
    # print(f"[INFO] {returncode=}")
    # return returncode  # 0 is good

    result = subprocess.run(command, shell=True, text=True, capture_output=True, timeout=timeout, cwd=cwd)

    with open(save_dir / "execute_out.txt", 'w') as f:
        f.write(result.stdout)
    with open(save_dir / "execute_err.txt", 'w') as f:
        f.write(result.stderr)

    print(f"[INFO] {result.returncode=}")
    if print_stdout:
        print(result.stdout)
    if print_stderr:
        print(result.stderr)
    return result.returncode  # 0 is good


def execute_command_retries(command: str, save_dir: str, retries=3, timeout=30):
    attempt = 0
    while attempt < retries:
        try:
            print(f"[INFO] Attempt {attempt=} of {retries=}")
            return execute_command(command=command, save_dir=save_dir, timeout=timeout)
        except subprocess.TimeoutExpired:
            attempt += 1
            time.sleep(1)  # Optional: Wait a bit before retrying
        except Exception as e:
            print(f"[ERROR] An error occurred: {e}, {command=}")
            import ipdb; ipdb.set_trace()
            break
    print(f"[ERROR] Failed to execute {command=} after {retries} attempts.")
    return None

from .app import run_api
import debugpy
import os
if __name__ == "__main__":
    # debugpy.listen(("0.0.0.0",8004))
    # print("123")
    # debugpy.wait_for_client()    
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    run_api()
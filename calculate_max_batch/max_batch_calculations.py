import sys
sys.path.append("..")
from batchflow import Pipeline, D, B, V, C, R, P
from batchflow.opensets import Imagenette160
from batchflow.models.torch import UNet
from batchflow import GPUMemoryMonitor
from batchflow.models.torch import EncoderDecoder
import torch
import numpy as np
import nvidia_smi
import time

def get_mem_info(device_id):
    gpu_list = [device_id]
    nvidia_smi.nvmlInit()
    handle = [nvidia_smi.nvmlDeviceGetHandleByIndex(i) for i in gpu_list]
    res = [nvidia_smi.nvmlDeviceGetMemoryInfo(item) for item in handle]
    res = [100 * item.used / item.total for item in res]
    nvidia_smi.nvmlShutdown()
    return res[0]

# def get_mem_info_with_torch(device_id):
#     t = torch.cuda.get_device_properties(device_id).total_memory
#     r = torch.cuda.memory_reserved(device_id) 
#     a = torch.cuda.memory_allocated(device_id)
#     f = r-a  # free inside reserved
#     return 

def get_run_mem(dataset, device_id, model_config, train_pipeline, batch_size=16, n_iters=50):
    with GPUMemoryMonitor(gpu_list=[device_id]) as monitor:
        torch.cuda.empty_cache()
        #time.sleep(60)
        train_pipeline.run(batch_size, n_iters=n_iters, bar=True)
    return np.max(monitor.data)

def get_max_batch_size(dataset, device_id, model_config, train_pipeline, init_batch_size, n_iters):
    first_run_memory = get_run_mem(dataset, device_id, model_config, train_pipeline, batch_size=init_batch_size, n_iters=n_iters)
    second_run_memory = get_run_mem(dataset, device_id, model_config, train_pipeline, batch_size=2*init_batch_size, n_iters=n_iters)
    max_batch_size = init_batch_size * (100 - 2 * first_run_memory + second_run_memory)/(second_run_memory - first_run_memory)
    return max_batch_size

def main():
    device_id = 4
    print("Used: ", get_mem_info(device_id))
    dataset = Imagenette160(bar=True)
    

    model_config = dict(model = UNet)
    model_config['device'] = f'cuda:{device_id}'
    model_config['loss'] = 'mse'

    train_pipeline = (dataset.train.p
                    .crop(shape=(160, 160), origin='center')
                    .init_variable('loss_history', [])
                    .to_array(channels='first', dtype=np.float32)
                    .multiply(1./255)
                    .init_model('dynamic', UNet, 'unet',
                                config=model_config)
                    .train_model('unet', B.images, B.images, 
                                 fetches='loss', save_to=V('loss_history', mode='a'), use_lock=True)
    )

#     init_batch_size = 2
#     n_iters = 50
#     batch_size = init_batch_size
#     torch.cuda.empty_cache()
#     first_run_memory = get_run_mem(dataset, device_id, model_config, train_pipeline, batch_size=batch_size, n_iters=n_iters)
#     torch.cuda.empty_cache()
#     for i in range(1, 6):
#         init_batch_size = pow(2,(i-1))*batch_size
#         second_run_memory = get_run_mem(dataset, device_id, model_config, train_pipeline, batch_size=pow(2,i)*batch_size, n_iters=n_iters)
#         print("Batches: ",  pow(2,(i-1))*batch_size,  pow(2,i)*batch_size)
#         print(first_run_memory, second_run_memory)
#         print("Max batch size:", init_batch_size * (100 - 2 * first_run_memory + second_run_memory)/(second_run_memory - first_run_memory))
#         first_run_memory = second_run_memory

    n_iters = 50
    batch_size = 78
    second_run_memory = get_run_mem(dataset, device_id, model_config, train_pipeline, batch_size=batch_size, n_iters=n_iters)
    print(second_run_memory)

        

    
if __name__ == "__main__":
    main()
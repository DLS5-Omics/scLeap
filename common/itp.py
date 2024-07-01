from tabulate import tabulate

data = [
    ["itpa100cl", "msrhyper", "card: A100, gpus_per_vm: 8, ib: yes, gpu_mem: 40GB"],
    ["itphyperdgx2cl1", "hai1", "card: V100, gpus_per_vm: 16, ib: yes, gpu_mem: 32GB"],
    ["itphyperdgx2cl1", "msrhyper", "card: V100, gpus_per_vm: 16, ib: yes, gpu_mem: 32GB"],
    ["itphyperbj1cl1", "msrhyper", "card: V100, gpus_per_vm: 16, ib: yes, gpu_mem: 32GB"],
    ["itphyperdellcl1", "msrhyper", "card: V100, gpus_per_vm: 4, ib: yes, gpu_mem: 32GB"],
    ["itphyperdgxcl1", "msrhyper", "card: V100, gpus_per_vm: 16, ib: yes, gpu_mem: 32GB"],
    ["itplabrr1cl1", "mprr3", "card: V100, gpus_per_vm: 8, ib: yes, gpu_mem: 32GB"],
    ["itplabrr1cl1", "resrchvc", "card: V100, gpus_per_vm: 8, ib: yes, gpu_mem: 32GB"],
    ["ms-shared", "MS-Shared", "card: A100/P100/P40/V100"],
]


def show_itp_list():
    print(tabulate(data, showindex=True, tablefmt="orgtbl"))


def choose_target():
    show_itp_list()
    idx = int(input("Choose target: "))
    assert 0 <= idx < len(data), "invalid target"
    return data[idx][0], data[idx][1]

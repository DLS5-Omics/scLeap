from tabulate import tabulate

data = [
    ["msroctovc", "V100 16G"],
    ["msrresrchvc", "V100 16G, A100 80G"],
    ["msrai4svc1", "A100 80G"],
    ["msrai4svc3", "A100 80G"],
]


def show_sing_list():
    print(tabulate(data, showindex=True, tablefmt="orgtbl"))


def choose_target():
    show_sing_list()
    idx = int(input("Choose target: "))
    assert 0 <= idx < len(data), "invalid target"
    return data[idx][0]

import os


def get_project_root() -> str:
    """获取项目根目录，兼容原 code/app.py 逻辑。

    优先以当前工作目录为基准：若 cwd 名称为 "code"，则返回上一层；否则返回 cwd。
    """
    current_dir = os.getcwd()
    if os.path.basename(current_dir) == "code":
        return os.path.dirname(current_dir)
    return current_dir


def get_data_paths(root: str) -> dict:
    """返回数据相关路径。

    包含：知识库 PDF 目录与 INDEX 目录。
    """
    return {
        "knowledge_dir": os.path.join(root, "data", "file"),
        "index_dir": os.path.join(root, "data", "INDEX"),
    }



import os


def main():
    problems = []

    # 入口存在性
    if not os.path.exists("src/app/app.py"):
        problems.append("缺少入口 src/app/app.py")

    # 关键模块文件
    for p in [
        "src/ingestion/loader.py",
        "src/index/vector_store.py",
        "src/retrieval/rag_chain.py",
        "src/reranker/model.py",
        "src/reranker/retriever.py",
        "src/common/paths.py",
        "src/common/logging.py",
    ]:
        if not os.path.exists(p):
            problems.append(f"缺少关键文件: {p}")

    # 测试资源目录
    for d in [
        "tests/resources/ingestion",
        "tests/resources/index",
        "tests/resources/retrieval",
        "tests/resources/reranker",
    ]:
        if not os.path.exists(d):
            problems.append(f"缺少测试资源目录: {d}")

    if problems:
        print("自检发现问题:")
        for m in problems:
            print(" - ", m)
        raise SystemExit(1)
    print("自检通过：基础结构完整。")


if __name__ == "__main__":
    main()



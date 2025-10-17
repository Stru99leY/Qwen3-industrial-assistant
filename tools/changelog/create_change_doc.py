import argparse
import datetime
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True, help="模块根目录，例如 src/reranker")
    parser.add_argument("--type", required=True, choices=["新增", "修改", "删除"])
    parser.add_argument("--file", required=True, help="本次变更的目标文件名，例如 model.py")
    parser.add_argument("--reason", required=True, help="变更原因")
    parser.add_argument("--impact", required=True, help="影响范围与测试要求")
    args = parser.parse_args()

    date = datetime.datetime.now().strftime("%Y%m%d")
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    changes_dir = os.path.join(args.module, "changes")
    os.makedirs(changes_dir, exist_ok=True)

    out_name = f"{date}_{args.type}_{args.file}.md"
    out_path = os.path.join(changes_dir, out_name)

    content = (
        "# 变更记录\n"
        f"- 变更时间: {now_str}\n"
        f"- 变更类型: {args.type}\n"
        f"- 变更文件: {args.file}\n"
        f"- 变更内容: {args.type} {args.file}\n"
        f"- 变更原因: {args.reason}\n"
        f"- 影响范围: {args.impact}\n"
    )

    # 如果已存在同名文件，则采用追加写入，保留历史记录
    write_mode = "a" if os.path.exists(out_path) else "w"
    prefix = "\n\n" if write_mode == "a" else ""
    with open(out_path, write_mode, encoding="utf-8") as f:
        f.write(prefix + content)

    print(f"已生成: {out_path}")


if __name__ == "__main__":
    main()



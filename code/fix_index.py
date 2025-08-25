# fix_index.py
"""
索引修复工具
用于解决FAISS索引维度不匹配等问题
"""

import os
import shutil
import sys
from pathlib import Path

def check_index_status():
    """检查索引状态"""
    INDEX_PATH = "data/INDEX"
    
    print("🔍 检查索引状态...")
    print(f"索引路径: {os.path.abspath(INDEX_PATH)}")
    
    if os.path.exists(INDEX_PATH):
        print("✅ 索引文件存在")
        
        # 检查索引文件大小
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(INDEX_PATH):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                total_size += file_size
                file_count += 1
                print(f"  - {file}: {file_size} bytes")
        
        print(f"📊 索引统计: {file_count} 个文件, 总大小: {total_size / 1024 / 1024:.2f} MB")
        
        return True
    else:
        print("❌ 索引文件不存在")
        return False

def backup_index():
    """备份现有索引"""
    INDEX_PATH = "data/INDEX"
    BACKUP_PATH = "data/INDEX_backup"
    
    if not os.path.exists(INDEX_PATH):
        print("❌ 没有索引文件需要备份")
        return False
    
    try:
        if os.path.exists(BACKUP_PATH):
            shutil.rmtree(BACKUP_PATH)
        
        shutil.copytree(INDEX_PATH, BACKUP_PATH)
        print(f"✅ 索引已备份到: {os.path.abspath(BACKUP_PATH)}")
        return True
    except Exception as e:
        print(f"❌ 备份失败: {e}")
        return False

def delete_index():
    """删除现有索引"""
    INDEX_PATH = "data/INDEX"
    
    if not os.path.exists(INDEX_PATH):
        print("❌ 索引文件不存在，无需删除")
        return True
    
    try:
        shutil.rmtree(INDEX_PATH)
        print("✅ 索引文件已删除")
        return True
    except Exception as e:
        print(f"❌ 删除失败: {e}")
        return False

def restore_index():
    """恢复备份的索引"""
    INDEX_PATH = "data/INDEX"
    BACKUP_PATH = "data/INDEX_backup"
    
    if not os.path.exists(BACKUP_PATH):
        print("❌ 没有找到备份文件")
        return False
    
    try:
        if os.path.exists(INDEX_PATH):
            shutil.rmtree(INDEX_PATH)
        
        shutil.copytree(BACKUP_PATH, INDEX_PATH)
        print("✅ 索引已从备份恢复")
        return True
    except Exception as e:
        print(f"❌ 恢复失败: {e}")
        return False

def main():
    """主函数"""
    print("🔧 FAISS索引修复工具")
    print("=" * 50)
    
    while True:
        print("\n请选择操作:")
        print("1. 检查索引状态")
        print("2. 备份现有索引")
        print("3. 删除现有索引")
        print("4. 恢复备份索引")
        print("5. 退出")
        
        choice = input("\n请输入选项 (1-5): ").strip()
        
        if choice == "1":
            check_index_status()
        
        elif choice == "2":
            backup_index()
        
        elif choice == "3":
            confirm = input("⚠️  确定要删除索引吗？这将导致需要重新构建知识库 (y/N): ").strip().lower()
            if confirm == 'y':
                delete_index()
            else:
                print("操作已取消")
        
        elif choice == "4":
            confirm = input("⚠️  确定要恢复备份索引吗？这将覆盖当前索引 (y/N): ").strip().lower()
            if confirm == 'y':
                restore_index()
            else:
                print("操作已取消")
        
        elif choice == "5":
            print("👋 再见！")
            break
        
        else:
            print("❌ 无效选项，请重新选择")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序出错: {e}")
        import traceback
        traceback.print_exc()

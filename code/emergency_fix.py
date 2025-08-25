# emergency_fix.py
"""
紧急修复脚本
直接删除FAISS索引文件，解决维度不匹配问题
"""

import os
import shutil
import sys

def emergency_delete_index():
    """紧急删除索引文件"""
    INDEX_PATH = "data/INDEX"
    
    print("🚨 紧急修复：删除FAISS索引文件")
    print("=" * 50)
    
    if os.path.exists(INDEX_PATH):
        try:
            # 备份索引（可选）
            backup_path = f"{INDEX_PATH}_backup_{int(time.time())}"
            print(f"📦 备份索引到: {backup_path}")
            shutil.copytree(INDEX_PATH, backup_path)
            
            # 删除索引
            print(f"🗑️  删除索引: {INDEX_PATH}")
            shutil.rmtree(INDEX_PATH)
            
            print("✅ 索引删除成功！")
            print("💡 现在请重新运行 streamlit run app.py")
            print(f"💾 备份保存在: {backup_path}")
            
        except Exception as e:
            print(f"❌ 删除失败: {e}")
            return False
    else:
        print("ℹ️  索引文件不存在，无需删除")
    
    return True

def force_rebuild():
    """强制重建索引"""
    print("\n🔧 强制重建索引")
    print("=" * 30)
    
    # 检查PDF文件
    pdf_dir = "data/file"
    if os.path.exists(pdf_dir):
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        print(f"📄 找到 {len(pdf_files)} 个PDF文件:")
        for pdf in pdf_files:
            print(f"  - {pdf}")
    else:
        print("❌ PDF目录不存在: data/file")
        return False
    
    print("\n💡 现在请运行以下命令重建索引:")
    print("streamlit run app.py")
    
    return True

if __name__ == "__main__":
    import time
    
    print("🔧 FAISS索引紧急修复工具")
    print("用于解决 'assert d == self.d' 错误")
    print()
    
    try:
        # 删除索引
        if emergency_delete_index():
            # 检查PDF文件
            force_rebuild()
        
        print("\n🎯 修复完成！请重新启动应用。")
        
    except KeyboardInterrupt:
        print("\n\n👋 操作被用户中断")
    except Exception as e:
        print(f"\n❌ 修复过程中出错: {e}")
        import traceback
        traceback.print_exc()

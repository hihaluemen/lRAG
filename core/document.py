from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import uuid


@dataclass(frozen=True)
class Document:
    """文档基类,用于在系统中传递数据"""
    content: str  # 文档内容
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    id: str = field(default_factory=lambda: str(uuid.uuid4()))  # 文档ID
    
    def __hash__(self):
        """使用id作为哈希值"""
        return hash(self.id)
    
    def __eq__(self, other):
        """基于id判断相等性"""
        if not isinstance(other, Document):
            return False
        return self.id == other.id
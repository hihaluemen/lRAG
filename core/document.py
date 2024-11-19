from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class Document:
    """文档基类,用于在系统中传递数据"""
    content: str  # 文档内容
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    id: Optional[str] = None  # 文档ID
    
    def __post_init__(self):
        if self.id is None:
            import uuid
            self.id = str(uuid.uuid4())
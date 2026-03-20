"""
Study Configuration Base Class

每个 study 都应该有一个独立的配置类，完全封装：
1. Trials 生成（从 specification 到具体实验试次）
2. Prompt 构建（使用 PromptBuilder）
3. 结果聚合（计算描述性统计、效应量等）
4. 自定义评分逻辑（可选）

这样每个 study 的逻辑完全独立，易于维护和扩展。
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.agents.prompt_builder import PromptBuilder


class BaseStudyConfig(ABC):
    """
    Study 配置基类
    
    每个 study 继承这个类并实现：
    - create_trials(): 生成实验 trials
    - aggregate_results(): 聚合结果（可选，有默认实现）
    - custom_scoring(): 自定义评分逻辑（可选）
    """
    
    prompt_builder_class = PromptBuilder  # 默认使用基类，子类可覆盖
    
    def __init__(self, study_path: Path, specification: Dict[str, Any]):
        """
        Args:
            study_path: 研究根目录路径 (e.g., data/studies/study_003/); 数据在 source/ 下
            specification: 研究 specification.json 内容
        """
        self.study_path = Path(study_path)
        self.source_path = self.study_path / "source"
        self.specification = specification
        self.study_id = specification["study_id"]
        
        # 使用指定的类初始化 prompt builder（传入 source 目录以便读取 specification 与 materials）
        self.prompt_builder = self.prompt_builder_class(self.source_path)

    def load_material(self, sub_study_id: str) -> Dict[str, Any]:
        """从 materials 目录加载指定 sub_study 的 JSON 文件"""
        file_path = self.source_path / "materials" / f"{sub_study_id}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Material not found: {file_path}")
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                return json.load(f)
        except UnicodeDecodeError as e:
            # Try to detect encoding and provide helpful error
            import chardet
            with open(file_path, "rb") as f:
                raw = f.read()
                detected = chardet.detect(raw)
            raise UnicodeDecodeError(
                'utf-8', raw, e.start, e.end,
                f"File encoding issue. Detected: {detected.get('encoding', 'unknown')} "
                f"(confidence: {detected.get('confidence', 0):.2f}). "
                f"Please ensure the file is UTF-8 encoded."
            )

    def load_metadata(self) -> Dict[str, Any]:
        """加载 metadata.json"""
        file_path = self.source_path / "metadata.json"
        with open(file_path, "r", encoding='utf-8') as f:
            return json.load(f)

    def load_specification(self) -> Dict[str, Any]:
        """加载 specification.json"""
        file_path = self.source_path / "specification.json"
        with open(file_path, "r", encoding='utf-8') as f:
            return json.load(f)

    def load_ground_truth(self) -> Dict[str, Any]:
        """加载 ground_truth.json"""
        file_path = self.source_path / "ground_truth.json"
        with open(file_path, "r", encoding='utf-8') as f:
            return json.load(f)

    def extract_numeric(self, text: str, default: float = 0.0) -> float:
        """从文本中提取第一个数字（支持负数和小数）"""
        if text is None: return default
        import re
        match = re.search(r"(-?\d+\.?\d*)", str(text))
        return float(match.group(1)) if match else default

    def extract_choice(self, text: str, options: List[str] = None) -> Optional[int]:
        """从文本中提取选项索引 (0, 1, 2...)"""
        if text is None: return None
        import re
        text_s = str(text).strip()
        
        # 1. 尝试匹配选项文本 (如果提供了 options)
        if options:
            for i, opt in enumerate(options):
                if opt.lower() in text_s.lower():
                    return i
        
        # 2. 尝试匹配单字母选项，如 "A", "Choice A", "(A)"
        match = re.search(r"\b([A-Z])\b", text_s.upper())
        if match:
            # A->0, B->1...
            return ord(match.group(1)) - ord('A')
            
        return None
    
    @abstractmethod
    def create_trials(self, n_trials: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        根据 specification 生成实验 trials
        
        Args:
            n_trials: 试次数量（None = 使用 specification 中的默认值）
        
        Returns:
            List of trial dictionaries，每个包含：
            - trial_number: int
            - study_type: str (e.g., "framing_effect")
            - trial_type: str (e.g., "practice", "critical", "neutral")
            - 其他 study-specific 字段
        """
        raise NotImplementedError
    
    def get_prompt_builder(self) -> PromptBuilder:
        """获取 prompt builder"""
        return self.prompt_builder
    
    def get_instructions(self) -> str:
        """获取实验说明"""
        return self.prompt_builder.get_instructions()
    
    def aggregate_results(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        聚合实验结果
        
        默认实现：直接返回 ParticipantPool.run_experiment() 的结果
        子类可以重写以添加自定义统计分析
        
        Args:
            raw_results: ParticipantPool.run_experiment() 返回的原始结果
        
        Returns:
            聚合后的结果字典，格式：
            {
                "descriptive_statistics": {...},
                "inferential_statistics": {...},
                "individual_data": [...],
                "raw_responses": [...]
            }
        """
        return raw_results
    
    def custom_scoring(
        self, 
        results: Dict[str, Any], 
        ground_truth: Dict[str, Any]
    ) -> Optional[Dict[str, float]]:
        """
        自定义评分逻辑（可选）
        
        如果 study 需要特殊的评分逻辑，重写此方法。
        
        Args:
            results: aggregate_results() 返回的结果
            ground_truth: ground_truth.json 内容
        
        Returns:
            None（使用默认 Scorer）或自定义分数字典：
            {
                "test_name_1": 0.8,
                "test_name_2": 0.6,
                ...
            }
        """
        return None
    
    def get_n_participants(self) -> int:
        """从 specification 获取参与者数量"""
        return self.specification["participants"]["n"]
    
    def get_study_type(self) -> str:
        """获取研究类型"""
        return self.specification.get("study_type", self.study_id)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(study_id='{self.study_id}')"


class StudyConfigRegistry:
    """
    Study 配置注册表
    
    自动发现和加载 study 配置类
    """
    
    _configs: Dict[str, type] = {}
    
    @classmethod
    def register(cls, study_id: str):
        """
        装饰器：注册 study 配置类
        
        Usage:
            @StudyConfigRegistry.register("study_003")
            class Study003Config(BaseStudyConfig):
                ...
        """
        def decorator(config_class):
            cls._configs[study_id] = config_class
            return config_class
        return decorator
    
    @classmethod
    def get_config_class(cls, study_id: str) -> Optional[type]:
        """获取配置类"""
        return cls._configs.get(study_id)
    
    @classmethod
    def create_config(
        cls, 
        study_id: str, 
        study_path: Path, 
        specification: Dict[str, Any]
    ) -> Optional[BaseStudyConfig]:
        """
        创建配置实例
        
        Args:
            study_id: 研究 ID
            study_path: 研究目录
            specification: specification.json 内容
        
        Returns:
            配置实例，如果未找到配置类则返回 None
        """
        config_class = cls.get_config_class(study_id)
        if config_class:
            return config_class(study_path, specification)
        return None
    
    @classmethod
    def list_registered_studies(cls) -> List[str]:
        """列出所有已注册的 study IDs"""
        return list(cls._configs.keys())


def get_study_config(
    study_id: str,
    study_path: Path,
    specification: Dict[str, Any]
) -> BaseStudyConfig:
    """
    从 study 目录加载 config.py 并创建配置实例。

    Args:
        study_id: 研究 ID (e.g., "study_003")
        study_path: 研究根目录 (e.g. data/studies/study_003/)，config 在 scripts/config.py
        specification: specification.json 内容

    Returns:
        Study 配置实例
    """
    import importlib.util
    import sys

    config_path = Path(study_path) / "scripts" / "config.py"
    if not config_path.exists():
        raise ValueError(f"Config not found: {config_path}")

    module_name = f"{study_id}_config"
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    module = importlib.util.module_from_spec(spec)
    if module_name not in sys.modules:
        sys.modules[module_name] = module
    spec.loader.exec_module(module)

    config_class = None
    for obj in vars(module).values():
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseStudyConfig)
            and obj is not BaseStudyConfig
        ):
            config_class = obj
            break
    if config_class is None:
        raise ValueError(f"No BaseStudyConfig subclass found in {config_path}")
    return config_class(Path(study_path), specification)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/11 18:45
@Author  : alexanderwu
@File    : action_node.py

NOTE: You should use typing.List instead of list to do type annotation. Because in the markdown extraction process,
  we can use typing to extract the type of the node, but we cannot use built-in list to extract.
"""
import json
import re
import typing
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel, Field, create_model, model_validator
from tenacity import retry, stop_after_attempt, wait_random_exponential

from maas.actions.action_outcls_registry import register_action_outcls
from maas.const import USE_CONFIG_TIMEOUT
from maas.llm import BaseLLM
from maas.logs import logger
from maas.provider.postprocess.llm_output_postprocess import llm_output_postprocess
from maas.utils.common import OutputParser, general_after_log
from maas.utils.human_interaction import HumanInteraction
from maas.utils.sanitize import sanitize
from typing import Type


class ReviewMode(Enum):
    HUMAN = "human"
    AUTO = "auto"


class ReviseMode(Enum):
    HUMAN = "human"  # human revise
    HUMAN_REVIEW = "human_review"  # human-review and auto-revise
    AUTO = "auto"  # auto-review and auto-revise


TAG = "CONTENT"


class FillMode(Enum):
    CODE_FILL = "code_fill"
    XML_FILL = "xml_fill"
    SINGLE_FILL = "single_fill"


LANGUAGE_CONSTRAINT = "Language: Please use the same language as Human INPUT."
FORMAT_CONSTRAINT = f"Format: output wrapped inside [{TAG}][/{TAG}] like format example, nothing else."


SIMPLE_TEMPLATE = """
## context
{context}

-----

## format example
{example}

## nodes: "<node>: <type>  # <instruction>"
{instruction}

## constraint
{constraint}

## action
Follow instructions of nodes, generate output and make sure it follows the format example.
"""

REVIEW_TEMPLATE = """
## context
Compare the key's value of nodes_output and the corresponding requirements one by one. If a key's value that does not match the requirement is found, provide the comment content on how to modify it. No output is required for matching keys.

### nodes_output
{nodes_output}

-----

## format example
[{tag}]
{{
    "key1": "comment1",
    "key2": "comment2",
    "keyn": "commentn"
}}
[/{tag}]

## nodes: "<node>: <type>  # <instruction>"
- key1: <class 'str'> # the first key name of mismatch key
- key2: <class 'str'> # the second key name of mismatch key
- keyn: <class 'str'> # the last key name of mismatch key

## constraint
{constraint}

## action
Follow format example's {prompt_schema} format, generate output and make sure it follows the format example.
"""

REVISE_TEMPLATE = """
## context
change the nodes_output key's value to meet its comment and no need to add extra comment.

### nodes_output
{nodes_output}

-----

## format example
{example}

## nodes: "<node>: <type>  # <instruction>"
{instruction}

## constraint
{constraint}

## action
Follow format example's {prompt_schema} format, generate output and make sure it follows the format example.
"""


def dict_to_markdown(d, prefix="- ", kv_sep="\n", postfix="\n"):
    markdown_str = ""
    for key, value in d.items():
        markdown_str += f"{prefix}{key}{kv_sep}{value}{postfix}"
    return markdown_str


class ActionNode:
    """ActionNode is a tree of nodes."""

    schema: str  # raw/json/markdown, default: ""

    # Action Context
    context: str  # all the context, including all necessary info
    llm: BaseLLM  # LLM with aask interface
    children: dict[str, "ActionNode"]

    # Action Input
    key: str  # Product Requirement / File list / Code
    func: typing.Callable  
    params: Dict[str, Type]  
    expected_type: Type  # such as str / int / float etc.
    instruction: str  # the instructions should be followed.
    example: Any  # example for In Context-Learning.

    # Action Output
    content: str
    instruct_content: BaseModel

    # For ActionGraph
    prevs: List["ActionNode"]  # previous nodes
    nexts: List["ActionNode"]  # next nodes

    def __init__(
        self,
        key: str,
        expected_type: Type,
        instruction: str,
        example: Any,
        content: str = "",
        children: dict[str, "ActionNode"] = None,
        schema: str = "",
    ):
        self.key = key
        self.expected_type = expected_type
        self.instruction = instruction
        self.example = example
        self.content = content
        self.children = children if children is not None else {}
        self.schema = schema
        self.prevs = []
        self.nexts = []
    
    @classmethod
    def from_pydantic(cls, pydantic_model):
        """
        Create an ActionNode from a Pydantic model class
        """
        # Extract required fields from the Pydantic model
        key = getattr(pydantic_model, 'key', None)
        expected_type = getattr(pydantic_model, 'expected_type', None)
        instruction = getattr(pydantic_model, 'instruction', None)
        example = getattr(pydantic_model, 'example', None)

        # If any are missing, raise an error so you know
        if None in [key, expected_type, instruction, example]:
            raise ValueError(
                f"Missing required fields in Pydantic model: key={key}, expected_type={expected_type}, instruction={instruction}, example={example}"
            )

        # Create the ActionNode instance
        instance = cls(
            key=key,
            expected_type=expected_type,
            instruction=instruction,
            example=example
        )
        return instance


    def __str__(self):
        return (
            f"{self.key}, {repr(self.expected_type)}, {self.instruction}, {self.example}"
            f", {self.content}, {self.children}"
        )

    def __repr__(self):
        return self.__str__()

    def add_prev(self, node: "ActionNode"):
        self.prevs.append(node)

    def add_next(self, node: "ActionNode"):
        self.nexts.append(node)

    def add_child(self, node: "ActionNode"):
        self.children[node.key] = node

    def get_child(self, key: str) -> Union["ActionNode", None]:
        return self.children.get(key, None)

    def add_children(self, nodes: List["ActionNode"]):
        for node in nodes:
            self.add_child(node)

    @classmethod
    def from_children(cls, key, nodes: List["ActionNode"]):
        obj = cls(key, str, "", "")
        obj.add_children(nodes)
        return obj

    def _get_children_mapping(self, exclude=None) -> Dict[str, Any]:
        exclude = exclude or []

        def _get_mapping(node: "ActionNode") -> Dict[str, Any]:
            mapping = {}
            for key, child in node.children.items():
                if key in exclude:
                    continue
                if child.children:
                    mapping[key] = _get_mapping(child)
                else:
                    mapping[key] = (child.expected_type, Field(default=child.example, description=child.instruction))
            return mapping

        return _get_mapping(self)

    def _get_self_mapping(self) -> Dict[str, Tuple[Type, Any]]:
        """get self key: type mapping"""
        return {self.key: (self.expected_type, ...)}

    def get_mapping(self, mode="children", exclude=None) -> Dict[str, Tuple[Type, Any]]:
        """get key: type mapping under mode"""
        if mode == "children" or (mode == "auto" and self.children):
            return self._get_children_mapping(exclude=exclude)
        return {} if exclude and self.key in exclude else self._get_self_mapping()

    @classmethod
    @register_action_outcls
    def create_model_class(cls, class_name: str, mapping: Dict[str, Tuple[Type, Any]]):

        def check_fields(cls, values):
            all_fields = set(mapping.keys())
            required_fields = set()
            for k, v in mapping.items():
                type_v, field_info = v
                if ActionNode.is_optional_type(type_v):
                    continue
                required_fields.add(k)

            missing_fields = required_fields - set(values.keys())
            if missing_fields:
                raise ValueError(f"Missing fields: {missing_fields}")

            unrecognized_fields = set(values.keys()) - all_fields
            if unrecognized_fields:
                logger.warning(f"Unrecognized fields: {unrecognized_fields}")
            return values

        validators = {"check_missing_fields_validator": model_validator(mode="before")(check_fields)}

        new_fields = {}
        for field_name, field_value in mapping.items():
            if isinstance(field_value, dict):
                nested_class_name = f"{class_name}_{field_name}"
                nested_class = cls.create_model_class(nested_class_name, field_value)
                new_fields[field_name] = (nested_class, ...)
            else:
                new_fields[field_name] = field_value

        new_class = create_model(class_name, __validators__=validators, **new_fields)
        return new_class

    def create_class(self, mode: str = "auto", class_name: str = None, exclude=None):
        class_name = class_name if class_name else f"{self.key}_AN"
        mapping = self.get_mapping(mode=mode, exclude=exclude)
        return self.create_model_class(class_name, mapping)

    def _create_children_class(self, exclude=None):

        class_name = f"{self.key}_AN"
        mapping = self._get_children_mapping(exclude=exclude)
        return self.create_model_class(class_name, mapping)

    def to_dict(self, format_func=None, mode="auto", exclude=None) -> Dict:

        nodes = self._to_dict(format_func=format_func, mode=mode, exclude=exclude)
        if not isinstance(nodes, dict):
            nodes = {self.key: nodes}
        return nodes

    def _to_dict(self, format_func=None, mode="auto", exclude=None) -> Dict:

        if format_func is None:
            format_func = lambda node: node.instruction

        formatted_value = format_func(self)
        if (mode == "children" or mode == "auto") and self.children:
            node_value = {}
        else:
            node_value = formatted_value

        if mode == "root":
            return {self.key: node_value}


        exclude = exclude or []
        for child_key, child_node in self.children.items():
            if child_key in exclude:
                continue

            child_dict = child_node._to_dict(format_func, mode, exclude)
            node_value[child_key] = child_dict

        return node_value

    def update_instruct_content(self, incre_data: dict[str, Any]):
        assert self.instruct_content
        origin_sc_dict = self.instruct_content.model_dump()
        origin_sc_dict.update(incre_data)
        output_class = self.create_class()
        self.instruct_content = output_class(**origin_sc_dict)

    def keys(self, mode: str = "auto") -> list:
        if mode == "children" or (mode == "auto" and self.children):
            keys = []
        else:
            keys = [self.key]
        if mode == "root":
            return keys

        for _, child_node in self.children.items():
            keys.append(child_node.key)
        return keys

    def compile_to(self, i: Dict, schema, kv_sep) -> str:
        if schema == "json":
            return json.dumps(i, indent=4, ensure_ascii=False)
        elif schema == "markdown":
            return dict_to_markdown(i, kv_sep=kv_sep)
        else:
            return str(i)

    def tagging(self, text, schema, tag="") -> str:
        if not tag:
            return text
        return f"[{tag}]\n{text}\n[/{tag}]"

    def _compile_f(self, schema, mode, tag, format_func, kv_sep, exclude=None) -> str:
        nodes = self.to_dict(format_func=format_func, mode=mode, exclude=exclude)
        text = self.compile_to(nodes, schema, kv_sep)
        return self.tagging(text, schema, tag)

    def compile_instruction(self, schema="markdown", mode="children", tag="", exclude=None) -> str:
        """compile to raw/json/markdown template with all/root/children nodes"""
        format_func = lambda i: f"{i.expected_type}  # {i.instruction}"
        return self._compile_f(schema, mode, tag, format_func, kv_sep=": ", exclude=exclude)

    def compile_example(self, schema="json", mode="children", tag="", exclude=None) -> str:
        """compile to raw/json/markdown examples with all/root/children nodes"""

        format_func = lambda i: i.example
        return self._compile_f(schema, mode, tag, format_func, kv_sep="\n", exclude=exclude)

    def compile(self, context, schema="json", mode="children", template=SIMPLE_TEMPLATE, exclude=[]) -> str:
        if schema == "raw":
            return f"{context}\n\n## Actions\n{LANGUAGE_CONSTRAINT}\n{self.instruction}"
        instruction = self.compile_instruction(schema="markdown", mode=mode, exclude=exclude)
        example = self.compile_example(schema=schema, tag=TAG, mode=mode, exclude=exclude)
        constraints = [LANGUAGE_CONSTRAINT, FORMAT_CONSTRAINT]
        constraint = "\n".join(constraints)

        prompt = template.format(
            context=context,
            example=example,
            instruction=instruction,
            constraint=constraint,
        )
        return prompt

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6),
        after=general_after_log(logger),
    )
    async def _aask_v1(
        self,
        prompt: str,
        output_class_name: str,
        output_data_mapping: dict,
        images: Optional[Union[str, list[str]]] = None,
        system_msgs: Optional[list[str]] = None,
        schema="markdown",  # compatible to original format
        timeout=USE_CONFIG_TIMEOUT,
    ) -> (str, BaseModel):
        """Use ActionOutput to wrap the output of aask"""
        content = await self.llm.aask(prompt, system_msgs, images=images, timeout=timeout)
        output_class = self.create_model_class(output_class_name, output_data_mapping)

        if schema == "json":
            parsed_data = llm_output_postprocess(
                output=content, schema=output_class.model_json_schema(), req_key=f"[/{TAG}]"
            )
        else:  # using markdown parser
            parsed_data = OutputParser.parse_data_with_mapping(content, output_data_mapping)

        instruct_content = output_class(**parsed_data)
        return content, instruct_content

    def get(self, key):
        return self.instruct_content.model_dump()[key]

    def set_recursive(self, name, value):
        setattr(self, name, value)
        for _, i in self.children.items():
            i.set_recursive(name, value)

    def set_llm(self, llm):
        self.set_recursive("llm", llm)

    def set_context(self, context):
        self.set_recursive("context", context)

    async def simple_fill(
        self, schema, mode, images: Optional[Union[str, list[str]]] = None, timeout=USE_CONFIG_TIMEOUT, exclude=None
    ):
        prompt = self.compile(context=self.context, schema=schema, mode=mode, exclude=exclude)
        if schema != "raw":
            mapping = self.get_mapping(mode, exclude=exclude)
            class_name = f"{self.key}_AN"
            content, scontent = await self._aask_v1(
                prompt, class_name, mapping, images=images, schema=schema, timeout=timeout
            )
            self.content = content
            self.instruct_content = scontent
        else:
            self.content = await self.llm.aask(prompt)
            self.instruct_content = None

        return self

    def get_field_name(self):
        """
        Get the field name from the Pydantic model associated with this ActionNode.
        """
        model_class = self.create_class()
        fields = model_class.model_fields

        if len(fields) == 1:
            return next(iter(fields))

        return self.key

    def get_field_names(self):
        """
        Get the field names associated with this ActionNode's Pydantic model.
        """
        model_class = self.create_class()
        return model_class.model_fields.keys()

    def get_field_types(self):
        """
        Get the field types associated with this ActionNode's Pydantic model.
        """
        model_class = self.create_class()
        return {field_name: field.annotation for field_name, field in model_class.model_fields.items()}

    def xml_compile(self, context):
        """
        Compile the prompt to make it easier for the model to understand the xml format.
        """
        field_names = self.get_field_names()
        examples = []
        for field_name in field_names:
            examples.append(f"<{field_name}>content</{field_name}>")

        example_str = "\n".join(examples)
        context += f"""
### Response format (must be strictly followed): All content must be enclosed in the given XML tags, ensuring each opening <tag> has a corresponding closing </tag>, with no incomplete or self-closing tags allowed.\n
{example_str}
"""
        return context

    def _unwrap_result(self, result: Any) -> Any:
        """
        Normalize the result dict before instantiating the Pydantic model.
        Handles wrappers like {'GenerateOp': '<json string>'} or {'GenerateOp': { ... }}.
        Returns either the inner parsed dict or the original result if no wrapper found.
        """
        try:
            if isinstance(result, dict) and len(result) == 1:
                only_key = next(iter(result))
                only_val = result[only_key]
                if isinstance(only_val, str):
                    try:
                        parsed = json.loads(only_val)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        return result
                if isinstance(only_val, dict):
                    return only_val
        except Exception:
            return result
        return result

    async def code_fill(
        self, context: str, function_name: Optional[str] = None, timeout: int = USE_CONFIG_TIMEOUT
    ) -> Dict[str, str]:
        field_name = self.get_field_name()
        prompt = context
        content = await self.llm.aask(prompt, timeout=timeout)
        extracted_code = sanitize(code=content, entrypoint=function_name)
        result = {field_name: extracted_code}
        return result

    async def single_fill(self, context: str, images: Optional[Union[str, list[str]]] = None) -> Dict[str, str]:
        field_name = self.get_field_name()
        prompt = context
        content = await self.llm.aask(prompt, images=images)
        result = {field_name: content}
        return result

    async def xml_fill(self, context: str, images: Optional[Union[str, list[str]]] = None) -> Dict[str, Any]:
        field_names = self.get_field_names()
        field_types = self.get_field_types()

        extracted_data: Dict[str, Any] = {}
        content = await self.llm.aask(context, images=images) 

        for field_name in field_names:
            # Extract the value for this field
            # Here we assume a simple JSON-like extraction from content
            try:
                parsed_json = json.loads(content)
                if field_name in parsed_json:
                    extracted_data[field_name] = parsed_json[field_name]
                else:
                    extracted_data[field_name] = None
            except json.JSONDecodeError:
                # Fallback: just return the raw content
                extracted_data[field_name] = content

        # Unwrap possible wrappers like {'GenerateOp': {...}} or {'GenerateOp': '...'}
        extracted_data = self._unwrap_result(extracted_data)

        return extracted_data

    async def fill(
        self,
        fill_mode: FillMode = FillMode.SINGLE_FILL,
        context: Optional[str] = None,
        images: Optional[Union[str, list[str]]] = None,
        timeout: int = USE_CONFIG_TIMEOUT,
        function_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fill this ActionNode with content based on the chosen fill_mode.
        """
        context = context or self.context

        if fill_mode == FillMode.SINGLE_FILL:
            result = await self.single_fill(context, images=images)
        elif fill_mode == FillMode.CODE_FILL:
            result = await self.code_fill(context, function_name=function_name, timeout=timeout)
        elif fill_mode == FillMode.XML_FILL:
            result = await self.xml_fill(context, images=images)
        else:
            raise ValueError(f"Unsupported fill mode: {fill_mode}")

        # Normalize the result before returning
        result = self._unwrap_result(result)

        return result

    @staticmethod
    def is_optional_type(type_hint):
        """Check if a type hint is Optional[...]"""
        origin = getattr(type_hint, "__origin__", None)
        return origin is Union and type(None) in type_hint.__args__

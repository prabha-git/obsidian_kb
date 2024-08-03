from langchain.retrievers.self_query.base import SelfQueryRetriever,_get_builtin_translator
from langchain_core.language_models import BaseLanguageModel
from typing import Any, List, Optional, Sequence, Tuple, Union, Dict
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain_core.prompts import BasePromptTemplate
from langchain_core.structured_query import Comparator, Operator,Visitor
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStore
from langchain.chains.query_constructor.base import StructuredQueryOutputParser

from langchain.chains.query_constructor.prompt import (
    USER_SPECIFIED_EXAMPLE_PROMPT,
    PREFIX_WITH_DATA_SOURCE,
    EXAMPLES_WITH_LIMIT,
    EXAMPLE_PROMPT,
    DEFAULT_PREFIX,
    SUFFIX_WITHOUT_DATA_SOURCE,
    DEFAULT_SUFFIX,
    )
import json


DEFAULT_SCHEMA = """\
<< Structured Request Schema >>
When responding use a markdown code snippet with a JSON object formatted in the following schema:

```json
{{{{
    "query": string \\ Original user query
    "filter": string \\ logical condition statement for filtering documents
}}}}
```

Original user query with any conditions in the filter should be mentioned in the query as well.

A logical condition statement is composed of one or more comparison and logical operation statements.

A comparison statement takes the form: `comp(attr, val)`:
- `comp` ({allowed_comparators}): comparator
- `attr` (string):  name of attribute to apply the comparison to
- `val` (string): is the comparison value

A logical operation statement takes the form `op(statement1, statement2, ...)`:
- `op` ({allowed_operators}): logical operator
- `statement1`, `statement2`, ... (comparison statements or logical operation statements): one or more statements to apply the operation to

Make sure that you only use the comparators and logical operators listed above and no others.
Make sure that filters only refer to attributes that exist in the data source.
Make sure that filters only use the attributed names with its function names if there are functions applied on them.
Make sure that filters only use format `YYYY-MM-DD` when handling date data typed values.
Make sure that filters take into account the descriptions of attributes and only make comparisons that are feasible given the type of data being stored.
Make sure that filters are only used as needed. If there are no filters that should be applied return "NO_FILTER" for the filter value.\
"""
FULL_ANSWER = """\
```json
{{
    "query": "What are songs by Taylor Swift or Katy Perry about teenage romance under 3 minutes long in the dance pop genre",
    "filter": "and(or(eq(\\"artist\\", \\"Taylor Swift\\"), eq(\\"artist\\", \\"Katy Perry\\")), lt(\\"length\\", 180), eq(\\"genre\\", \\"pop\\"))"
}}
```\
"""
SONG_DATA_SOURCE = """\
```json
{{
    "content": "Lyrics of a song",
    "attributes": {{
        "artist": {{
            "type": "string",
            "description": "Name of the song artist"
        }},
        "length": {{
            "type": "integer",
            "description": "Length of the song in seconds"
        }},
        "genre": {{
            "type": "string",
            "description": "The song genre, one of \"pop\", \"rock\" or \"rap\""
        }}
    }}
}}
```\
"""
NO_FILTER_ANSWER = """\
```json
{{
    "query": "",
    "filter": "NO_FILTER"
}}
```\
"""
DEFAULT_EXAMPLES = [
    {
        "i": 1,
        "data_source": SONG_DATA_SOURCE,
        "user_query": "What are songs by Taylor Swift or Katy Perry about teenage romance under 3 minutes long in the dance pop genre",
        "structured_request": FULL_ANSWER,
    },
    {
        "i": 2,
        "data_source": SONG_DATA_SOURCE,
        "user_query": "What are songs that were not published on Spotify",
        "structured_request": NO_FILTER_ANSWER,
    },
]

DEFAULT_SCHEMA_PROMPT = PromptTemplate.from_template(DEFAULT_SCHEMA)

def construct_examples(input_output_pairs: Sequence[Tuple[str, dict]]) -> List[dict]:
    """Construct examples from input-output pairs.

    Args:
        input_output_pairs: Sequence of input-output pairs.

    Returns:
        List of examples.
    """
    examples = []
    for i, (_input, output) in enumerate(input_output_pairs):
        structured_request = (
            json.dumps(output, indent=4).replace("{", "{{").replace("}", "}}")
        )
        example = {
            "i": i + 1,
            "user_query": _input,
            "structured_request": structured_request,
        }
        examples.append(example)
    return examples

def get_query_constructor_prompt(
    document_contents: str,
    attribute_info: Sequence[Union[AttributeInfo, dict]],
    *,
    examples: Optional[Sequence] = None,
    allowed_comparators: Sequence[Comparator] = tuple(Comparator),
    allowed_operators: Sequence[Operator] = tuple(Operator),
    enable_limit: bool = False,
    schema_prompt: Optional[BasePromptTemplate] = None,
    **kwargs: Any,
) -> BasePromptTemplate:
    """Create query construction prompt.

    Args:
        document_contents: The contents of the document to be queried.
        attribute_info: A list of AttributeInfo objects describing
            the attributes of the document.
        examples: Optional list of examples to use for the chain.
        allowed_comparators: Sequence of allowed comparators.
        allowed_operators: Sequence of allowed operators.
        enable_limit: Whether to enable the limit operator. Defaults to False.
        schema_prompt: Prompt for describing query schema. Should have string input
            variables allowed_comparators and allowed_operators.
        kwargs: Additional named params to pass to FewShotPromptTemplate init.

    Returns:
        A prompt template that can be used to construct queries.
    """
    default_schema_prompt = DEFAULT_SCHEMA_PROMPT
    schema_prompt = schema_prompt or default_schema_prompt
    attribute_str = _format_attribute_info(attribute_info)
    schema = schema_prompt.format(
        allowed_comparators=" | ".join(allowed_comparators),
        allowed_operators=" | ".join(allowed_operators),
    )
    if examples and isinstance(examples[0], tuple):
        examples = construct_examples(examples)
        example_prompt = USER_SPECIFIED_EXAMPLE_PROMPT
        prefix = PREFIX_WITH_DATA_SOURCE.format(
            schema=schema, content=document_contents, attributes=attribute_str
        )
        suffix = SUFFIX_WITHOUT_DATA_SOURCE.format(i=len(examples) + 1)
    else:
        examples = examples or (
            EXAMPLES_WITH_LIMIT if enable_limit else DEFAULT_EXAMPLES
        )
        example_prompt = EXAMPLE_PROMPT
        prefix = DEFAULT_PREFIX.format(schema=schema)
        suffix = DEFAULT_SUFFIX.format(
            i=len(examples) + 1, content=document_contents, attributes=attribute_str
        )
    return FewShotPromptTemplate(
        examples=list(examples),
        example_prompt=example_prompt,
        input_variables=["query"],
        suffix=suffix,
        prefix=prefix,
        **kwargs,
    )

def _format_attribute_info(info: Sequence[Union[AttributeInfo, dict]]) -> str:
    info_dicts = {}
    for i in info:
        i_dict = dict(i)
        info_dicts[i_dict.pop("name")] = i_dict
    return json.dumps(info_dicts, indent=4).replace("{", "{{").replace("}", "}}")

def load_query_constructor_runnable(
    llm: BaseLanguageModel,
    document_contents: str,
    attribute_info: Sequence[Union[AttributeInfo, dict]],
    *,
    examples: Optional[Sequence] = None,
    allowed_comparators: Sequence[Comparator] = tuple(Comparator),
    allowed_operators: Sequence[Operator] = tuple(Operator),
    enable_limit: bool = False,
    schema_prompt: Optional[BasePromptTemplate] = None,
    fix_invalid: bool = False,
    **kwargs: Any,
) -> Runnable:
    """Load a query constructor runnable chain.

    Args:
        llm: BaseLanguageModel to use for the chain.
        document_contents: Description of the page contents of the document to be
            queried.
        attribute_info: Sequence of attributes in the document.
        examples: Optional list of examples to use for the chain.
        allowed_comparators: Sequence of allowed comparators. Defaults to all
            Comparators.
        allowed_operators: Sequence of allowed operators. Defaults to all Operators.
        enable_limit: Whether to enable the limit operator. Defaults to False.
        schema_prompt: Prompt for describing query schema. Should have string input
            variables allowed_comparators and allowed_operators.
        fix_invalid: Whether to fix invalid filter directives by ignoring invalid
            operators, comparators and attributes.
        kwargs: Additional named params to pass to FewShotPromptTemplate init.

    Returns:
        A Runnable that can be used to construct queries.
    """
    prompt = get_query_constructor_prompt(
        document_contents,
        attribute_info,
        examples=examples,
        allowed_comparators=allowed_comparators,
        allowed_operators=allowed_operators,
        enable_limit=enable_limit,
        schema_prompt=schema_prompt,
        **kwargs,
    )
    allowed_attributes = []
    for ainfo in attribute_info:
        allowed_attributes.append(
            ainfo.name if isinstance(ainfo, AttributeInfo) else ainfo["name"]
        )
    output_parser = StructuredQueryOutputParser.from_components(
        allowed_comparators=allowed_comparators,
        allowed_operators=allowed_operators,
        allowed_attributes=allowed_attributes,
        fix_invalid=fix_invalid,
    )
    return prompt | llm | output_parser

QUERY_CONSTRUCTOR_RUN_NAME = "query_constructor"

class CustomSelfQueryRetriever(SelfQueryRetriever):

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        vectorstore: VectorStore,
        document_contents: str,
        metadata_field_info: Sequence[Union[AttributeInfo, dict]],
        structured_query_translator: Optional[Visitor] = None,
        chain_kwargs: Optional[Dict] = None,
        enable_limit: bool = False,
        use_original_query: bool = False,
        **kwargs: Any,
    ) -> "SelfQueryRetriever":
        if structured_query_translator is None:
            structured_query_translator = _get_builtin_translator(vectorstore)
        chain_kwargs = chain_kwargs or {}

        if (
            "allowed_comparators" not in chain_kwargs
            and structured_query_translator.allowed_comparators is not None
        ):
            chain_kwargs["allowed_comparators"] = (
                structured_query_translator.allowed_comparators
            )
        if (
            "allowed_operators" not in chain_kwargs
            and structured_query_translator.allowed_operators is not None
        ):
            chain_kwargs["allowed_operators"] = (
                structured_query_translator.allowed_operators
            )
        query_constructor = load_query_constructor_runnable(
            llm,
            document_contents,
            metadata_field_info,
            enable_limit=enable_limit,
            **chain_kwargs,
        )
        query_constructor = query_constructor.with_config(
            run_name=QUERY_CONSTRUCTOR_RUN_NAME
        )
        return cls(  # type: ignore[call-arg]
            query_constructor=query_constructor,
            vectorstore=vectorstore,
            use_original_query=use_original_query,
            structured_query_translator=structured_query_translator,
            **kwargs,
        )
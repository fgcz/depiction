import pytest
from snakemake_invoke.invoke.invoke_subprocess import InvokeSubprocess


@pytest.mark.parametrize(
    "input_args,expected",
    [
        pytest.param(["a"], "a", id="plain_string"),
        pytest.param(["a b"], "'a b'", id="string_with_space"),
        pytest.param(['"a b"'], "'\"a b\"'", id="string_with_double_quotes"),
        pytest.param(["a'b"], "'a'\"'\"'b'", id="string_with_single_quote"),
        pytest.param(['a"b'], "'a\"b'", id="string_with_double_quote"),
        pytest.param(["a", "a b", ""], "a 'a b' ''", id="list_of_strings"),
    ],
)
def test_args_to_shell_command(input_args, expected):
    result = InvokeSubprocess._args_to_shell_command(input_args)
    assert result == expected

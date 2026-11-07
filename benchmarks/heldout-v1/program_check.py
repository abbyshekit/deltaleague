"""Check generated GSM arithmetic in the pinned WASI sandbox, never host Python."""
import ast
import math

BAD_CALLS = {'open', 'exec', 'eval', '__import__', 'compile', 'input', 'breakpoint'}
BAD_BASES = {'os', 'sys', 'subprocess', 'socket', 'builtins'}


def veto_program(program):
    try:
        tree = ast.parse(program)
    except SyntaxError as e:
        return f'program syntax error: {e.msg}'
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name != 'math' for a in node.names):
            return 'program imports ' + ', '.join(a.name for a in node.names)
        if isinstance(node, ast.ImportFrom) and (node.module != 'math' or node.level):
            return 'program imports ' + '.' * node.level + (node.module or '')
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BAD_CALLS:
            return f'program calls {node.func.id}'
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in BAD_BASES:
            return f'program uses {node.value.id}.{node.attr}'
    return None



def run_program(program, numeric_answer):
    if not program.strip():
        return 'no program'
    if not math.isfinite(numeric_answer):
        return 'answer not finite'
    if (veto := veto_program(program)):
        return veto
    # This syntax screen is only a format policy. WASI enforces isolation even
    # when aliases or other Python features get around the screen.
    from delta_cli.public_eval.wasi import sandbox
    if not sandbox('assert 1 + 1 == 2')['passed']:
        raise RuntimeError('WASI sandbox preflight failed')
    code = program + '\n\nimport math as _delta_math\n_delta_value = float(solve())\n'
    code += f'assert _delta_math.isfinite(_delta_value) and abs(_delta_value - {numeric_answer!r}) < 1e-6\n'
    result = sandbox(code)
    if result['passed']:
        return None
    return 'program failed sandbox validation: ' + result['reason']

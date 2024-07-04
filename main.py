import os
import json
import inspect
import traceback
from matplotlib.colors import ListedColormap, Normalize
import tqdm

import arc_types
import constants
import dsl
import tests
import solvers
import matplotlib.pyplot as plt



def get_data(train=True):
    path = f'../ARC-AGI/data/{"training" if train else "evaluation"}'
    data = {}
    for fn in os.listdir(path):
        with open(f'{path}/{fn}') as f:
            data[fn.rstrip('.json')] = json.load(f)
    ast = lambda g: tuple(tuple(r) for r in g)
    return {
        'train': {k: [{
            'input': ast(e['input']),
            'output': ast(e['output']),
        } for e in v['train']] for k, v in data.items()},
        'test': {k: [{
            'input': ast(e['input']),
            'output': ast(e['output']),
        } for e in v['test']] for k, v in data.items()}
    }


def get_functions(path):
    """ returns a list of available functions """
    with open(path, 'r') as f:
        code = f.read()
    functions = []
    for row in code.split('\n'):
        if row.startswith('def '):
            function = row.split('def ')[1].split('(')[0]
            functions.append(function)
    return functions


def run_dsl_tests(dsl_module, test_module):
    """ test DSL primitives """
    dsl_functions = get_functions(dsl_module.__file__)
    test_functions = get_functions(test_module.__file__)
    expected = set([f'test_{f}' for f in dsl_functions])
    assert set(test_functions) == expected
    for fun in test_functions:
        getattr(test_module, fun)()


def test_solvers_formatting(solvers_module, dsl_module):
    """ tests the implementd solvers for formatting """
    with open('constants.py', 'r') as f:
        constants = [c.split(' = ')[0] for c in f.readlines() if ' = ' in c]
    definitions = {
        function: inspect.getsource(getattr(solvers_module, function)) \
            for function in get_functions(solvers_module.__file__)
    }
    dsl_interface = get_functions(dsl_module.__file__)
    n_correct = 0
    n = len(definitions)
    for key, definition in definitions.items():
        try:
            lines = definition.split('\n')
            assert lines[0] == f'def {key}(I):'
            assert lines[-1] == ''
            variables = set()
            calls = set()
            for line in lines[1:-2]:
                variable, call = line.lstrip().split(' = ')
                function, args = call.split('(')
                assert variable not in dsl_interface
                assert variable not in variables
                assert call not in calls
                variables.add(variable)
                calls.add(call)
                assert function in dsl_interface or function in variables
                assert args[-1] == ')'
                args = [args[:-1]] if ',' not in args else args[:-1].split(', ')
                for arg in args:
                    assert any([
                        arg in variables, arg in dsl_interface,
                        arg in constants, arg == 'I'
                    ])
            for v in variables:
                assert sum([
                    definition.count(vs) for vs in [
                        f'({v})', f'({v}, ', f', {v})',
                        f', {v}, ', f' {v} = ', f' {v}('
                    ]
                ]) > 1 or v == 'O'
            n_correct += 1
        except Exception as e:
            print(f'Error in {key}: {call}')
            print(traceback.format_exc())
            print(line.lstrip())
            pass
    print(f'{n_correct} out of {n} solvers formatted correctly.')

def plot_task(name, index, input, output, solved):
    """ plots a task """
    cmap = ListedColormap([
        '#000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ])
    norm = Normalize(vmin=0, vmax=9)
    figure, axes = plt.subplots(1, 3, figsize=(3 * 4, 8))
    figure.suptitle(f"Task {name} Iteration {index}", fontsize = 22)
    axes[0].imshow(input, cmap=cmap, norm=norm)
    axes[1].imshow(output, cmap=cmap, norm=norm)
    axes[2].imshow(solved, cmap=cmap, norm=norm)
    axes[0].set_title('Input')
    axes[0].axis('off')
    axes[1].set_title('Output')
    axes[1].axis('off')
    axes[2].set_title('Solved')
    axes[2].axis('off')
    plt.show()

def test_solvers_correctness(data, solvers_module):
    """ tests the implemented solvers for correctness """
    n_correct = 0
    n = len(data["train"])
    for key in data['train'].keys():
        task = data['train'][key] + data['test'][key]
        solver = getattr(solvers_module, f'solve_{key}')
        n_correct += 1
        i = 0
        for ex in task:
            solved = solver(ex['input'])
            if solved != ex['output']:
                plot_task(key, i, ex['input'], ex['output'], solved)
                n_correct -= 1
            i+=1
    print(f'{n_correct} out of {n} tasks solved correctly.')


def main():
    data = get_data(train=True)
    # run_dsl_tests(dsl, tests)
    test_solvers_formatting(solvers, dsl)
    test_solvers_correctness(data, solvers)


if __name__ == '__main__':
    main()

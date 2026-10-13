"""CORA integration contract with the pinned upstream MOISE Diagram package."""
from pathlib import Path
import shutil
import pytest
from moise_diagram import DiagramError, parse, to_dot, compile_file

SOURCE = Path(__file__).resolve().parents[1] / 'examples/moise/gouter.moise'


def test_example_semantics():
    model = parse(SOURCE.read_text())
    assert set(model.roles) >= {'preparateur', 'serveur'}
    assert len(model.schemes) == 1
    assert len(model.commits) == 2
    assert model.missions_of_goal('biscuits') == ['m_biscuits']
    assert 'soc' in model.ancestors('serveur')


def test_norm_matrix_and_combined_views():
    model = parse(SOURCE.read_text())
    matrix = to_dot(model, views=['ds'], ds_style='matrix', ds_inherited=True)
    assert 'm_verres' in matrix
    assert 'obl' in matrix
    combined = to_dot(model, views=['ss', 'fs'], key=True, ds_links=True)
    assert 'mx_deferred' in combined


def test_forbid_unauthorized_commitment():
    source = SOURCE.read_text().replace('commits lina m_verres', 'commits lina m_biscuits')
    with pytest.raises(DiagramError, match='cannot commit'):
        parse(source)


def test_forbid_goals_without_scheme():
    with pytest.raises(DiagramError, match='social scheme'):
        parse('@moise\ngoal orphan\n@end')


@pytest.mark.skipif(not (shutil.which('dot') and shutil.which('neato')), reason='Graphviz required')
@pytest.mark.parametrize('view,style', [('ss','list'),('fs','list'),('ds','matrix'),('oe','list')])
def test_render_each_view(tmp_path,view,style):
    out=tmp_path/f'{view}.svg'
    compile_file(SOURCE,out,views=[view],key=True,ds_style=style)
    assert '<svg' in out.read_text()


@pytest.mark.skipif(not (shutil.which('dot') and shutil.which('neato')), reason='Graphviz required')
def test_combined_pdf_and_compatibility_entry_point(tmp_path):
    from cora.moise_diagram import compile_file as compatibility_compile
    out=tmp_path/'organization.pdf'
    compatibility_compile(SOURCE,out,views=['ss','fs'],ds_links=True)
    assert out.read_bytes().startswith(b'%PDF')

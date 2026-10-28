"""Mixed k=2 gold-only prompt contract tests; no retrieval/model calls."""
import pytest
from tffv.experiments import m162b


def record(sid,error,gold,group):
    return {'sample_id':sid,'is_base_error':error,'gold_verdict':gold,'group_id':group,
      'claim':sid+' claim','base_prediction':{'reparsed_verdict':'Refuted'},
      'evidence':[{'passage_id':sid+'p','source_url':'https://example.org','rendered_text':'source evidence'}],
      'citations':[],'corrected_explanation':'MUST NOT APPEAR'}


def test_mixed_prompt_has_fixed_correct_then_wrong_gold_only_order():
    good=record('train:10',False,'Supported','source-good');bad=record('train:25',True,'Supported','source-bad')
    slate={'sample_id':'train:810','correct_id':'train:10','error_id':'train:25','correct_context':[{'source_url':'https://good','text':'good evidence'}],'error_context':[{'source_url':'https://bad','text':'bad evidence'}]}
    messages=m162b.mixed_messages('target',[{'text':'target evidence','source_url':'https://target'}],slate,{'records':[good,bad]})
    text=messages[1]['content']
    assert text.index('Base-correct')<text.index('Base-wrong')
    assert 'Previous Base verdict: [not provided]' in text
    assert 'MUST NOT APPEAR' not in text and 'incorrect for this historical example' not in text


def test_mixed_pair_rejects_label_or_group_leakage():
    good=record('good',False,'Supported','same');bad=record('bad',True,'Supported','same')
    class Contract: rows={'target':{'group_id':'target-group'}}
    with pytest.raises(ValueError,match='source groups overlap'):
        m162b.validate_pair({'sample_id':'target','correct_id':'good','error_id':'bad'},{'good':good,'bad':bad},Contract())
    bad['group_id']='other';bad['gold_verdict']='Refuted'
    with pytest.raises(ValueError,match='share source gold'):
        m162b.validate_pair({'sample_id':'target','correct_id':'good','error_id':'bad'},{'good':good,'bad':bad},Contract())
    bad['gold_verdict']='Supported';good['group_id']='target-group'
    with pytest.raises(ValueError,match='group leakage'):
        m162b.validate_pair({'sample_id':'target','correct_id':'good','error_id':'bad'},{'good':good,'bad':bad},Contract())

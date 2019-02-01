"""
@file
@brief Helpers to visualize a pipeline.
"""
import numpy
import pandas
from sklearn.base import TransformerMixin, ClassifierMixin, RegressorMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor


def _pipeline_info(pipe, data, context):
    """
    Internal function to convert a pipeline into
    some graph.
    """
    def _get_name(context, prefix='-v-'):
        sug = "%s%d" % (prefix, context['n'])
        while sug in context['names']:
            context['n'] += 1
            sug = "%s%d" % (prefix, context['n'])
        context['names'].add(sug)
        return sug

    if isinstance(pipe, Pipeline):
        infos = []
        for _, model in pipe.steps:
            info = _pipeline_info(model, data, context)
            data = info[-1]["outputs"]
            infos.extend(info)
        return infos
    elif isinstance(pipe, ColumnTransformer):
        infos = []
        outputs = []
        for _, model, vs in pipe.transformers:
            info = _pipeline_info(model, vs, context)
            new_outputs = []
            for o in info[-1]['outputs']:
                add = _get_name(context, prefix=o)
                outputs.append(add)
                new_outputs.append(add)
            info[-1]['outputs'] = new_outputs
            infos.extend(info)
        if len(pipe.transformers) > 1:
            infos.append({'name': 'union', 'outputs': [_get_name(context)], 'inputs': outputs,
                          'type': 'transform'})
        return infos
    elif isinstance(pipe, TransformedTargetRegressor):
        raise NotImplementedError(
            "Not yet implemented for TransformedTargetRegressor.")
    elif isinstance(pipe, FeatureUnion):
        raise NotImplementedError("Not yet implemented for FeatureUnion.")
    elif isinstance(pipe, TransformerMixin):
        info = {'name': pipe.__class__.__name__, 'type': 'transform'}
        if len(data) == 1:
            info['outputs'] = data
            info['inputs'] = data
            info = [info]
        else:
            info['inputs'] = [_get_name(context)]
            info['outputs'] = [_get_name(context)]
            info = [{'name': 'union', 'outputs': info['inputs'], 'inputs': data,
                     'type': 'transform'}, info]
        return info
    elif isinstance(pipe, ClassifierMixin):
        info = {'name': pipe.__class__.__name__, 'type': 'classifier'}
        exp = ['PredictedLabel', 'Probabilities']
        if len(data) == 1:
            info['outputs'] = exp
            info['inputs'] = data
            info = [info]
        else:
            info['outputs'] = exp
            info['inputs'] = [_get_name(context)]
            info = [{'name': 'union', 'outputs': info['inputs'], 'inputs': data,
                     'type': 'transform'}, info]
        return info
    elif isinstance(pipe, RegressorMixin):
        info = {'name': pipe.__class__.__name__, 'type': 'regressor'}
        exp = ['Prediction']
        if len(data) == 1:
            info['outputs'] = exp
            info['inputs'] = data
            info = [info]
        else:
            info['outputs'] = exp
            info['inputs'] = [_get_name(context)]
            info = [{'name': 'union', 'outputs': info['inputs'], 'inputs': data,
                     'type': 'transform'}, info]
        return info
    else:
        raise NotImplementedError(
            "Not yet implemented for {}.".format(type(pipe)))


def pipeline2dot(pipe, data, **params):
    """
    Exports a *scikit-learn* pipeline to
    :epkg:`DOT` language.

    @param      pipe        *scikit-learn* pipeline
    @param      data        training data as a dataframe or a numpy array,
                            or just a list with the variable names
    @param      params      additional params to draw the graph
    @return                 string
    """
    if isinstance(data, pandas.DataFrame):
        data = list(data.columns)
    elif isinstance(data, numpy.ndarray):
        if len(data.shape) != 2:
            raise NotImplementedError(
                "Unexpected training data dimension: {}.".format(data.shape))
        data = ['X[0-{}]'.format(data.shape[1])]
    elif not isinstance(data, list):
        raise TypeError("Unexpected data type: {}.".format(type(data)))

    exp = ["digraph{", "  orientation=portrait;"]
    fontsize = 8
    info = [dict(schema_after=data)]
    info.extend(_pipeline_info(pipe, data, context=dict(n=0, names=set(data))))
    columns = {}

    for i, line in enumerate(info):
        if i == 0:
            schema = line['schema_after']
            labs = []
            for c, col in enumerate(schema):
                columns[col] = 'sch0:f{0}'.format(c)
                labs.append("<f{0}> {1}".format(c, col))
            node = '  sch0[label="{0}",shape=record,fontsize={1}];'.format(
                "|".join(labs), params.get('fontsize', fontsize))
            exp.append(node)

        else:
            exp.append('')
            if line['type'] == 'transform':
                node = '  node{0}[label="{1}",shape=box,style="filled' \
                    ',rounded",color=cyan,fontsize={2}];'.format(
                        i, line['name'],
                        int(params.get('fontsize', fontsize) * 1.5))
            else:
                node = '  node{0}[label="{1}",shape=box,style="filled,' \
                    'rounded",color=yellow,fontsize={2}];'.format(
                        i, line['name'],
                        int(params.get('fontsize', fontsize) * 1.5))
            exp.append(node)

            for inp in line['inputs']:
                nc = columns[inp]
                edge = '  {0} -> node{1};'.format(nc, i)
                exp.append(edge)

            labs = []
            for c, out in enumerate(line['outputs']):
                columns[out] = 'sch{0}:f{1}'.format(i, c)
                labs.append("<f{0}> {1}".format(c, out))
            node = '  sch{0}[label="{1}",shape=record,fontsize={2}];'.format(
                i, "|".join(labs), params.get('fontsize', fontsize))
            exp.append(node)

            for out in line['outputs']:
                nc = columns[out]
                edge = '  node{1} -> {0};'.format(nc, i)
                exp.append(edge)

    exp.append('}')
    return "\n".join(exp)

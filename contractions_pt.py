#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__='thiagocastroferreira'


"""
Author: Thiago Castro Ferreira
Date: 28/05/2018
Description:
    Script for parsing and generating the contractions of the Portuguese language

PYTHON VERSION: 2.7
"""

import re

def parse(text):
    text = re.sub(r'( )(cum)([ ,;!?.])', r'\1 com um\3', text, flags=re.U)
    text = re.sub(r'(Cum)([ ,;!?.])', r'Com um\2', text, flags=re.U)

    text = re.sub(r'( )(do)([ ,;!?.])', r'\1de o\3', text, flags=re.U)
    text = re.sub(r'( )(da)([ ,;!?.])', r'\1de a\3', text, flags=re.U)
    text = re.sub(r'( )(dos)([ ,;!?.])', r'\1de os\3', text, flags=re.U)
    text = re.sub(r'( )(das)([ ,;!?.])', r'\1de as\3', text, flags=re.U)

    text = re.sub(r'(Do)([ ,;!?.])', r'De o\2', text, flags=re.U)
    text = re.sub(r'(Da)([ ,;!?.])', r'De a\2', text, flags=re.U)
    text = re.sub(r'(Dos)([ ,;!?.])', r'De os\2', text, flags=re.U)
    text = re.sub(r'(Das)([ ,;!?.])', r'De as\2', text, flags=re.U)

    text = re.sub(r'( )(dum)([ ,;!?.])', r'\1de um\3', text, flags=re.U)
    text = re.sub(r'( )(duma)([ ,;!?.])', r'\1de uma\3', text, flags=re.U)
    text = re.sub(r'( )(duns)([ ,;!?.])', r'\1de uns\3', text, flags=re.U)
    text = re.sub(r'( )(dumas)([ ,;!?.])', r'\1de umas\3', text, flags=re.U)

    text = re.sub(r'(Dum)([ ,;!?.])', r'De um\2', text, flags=re.U)
    text = re.sub(r'(Duma)([ ,;!?.])', r'De uma\2', text, flags=re.U)
    text = re.sub(r'(Duns)([ ,;!?.])', r'De uns\2', text, flags=re.U)
    text = re.sub(r'(Dumas)([ ,;!?.])', r'De umas\2', text, flags=re.U)

    text = re.sub(r'( )(dele)([ ,;!?.])', r'\1de ele\3', text, flags=re.U)
    text = re.sub(r'( )(dela)([ ,;!?.])', r'\1de ela\3', text, flags=re.U)
    text = re.sub(r'( )(deles)([ ,;!?.])', r'\1de eles\3', text, flags=re.U)
    text = re.sub(r'( )(delas)([ ,;!?.])', r'\1de elas\3', text, flags=re.U)

    text = re.sub(r'(Dele)([ ,;!?.])', r'De ele\2', text, flags=re.U)
    text = re.sub(r'(Dela)([ ,;!?.])', r'De ela\2', text, flags=re.U)
    text = re.sub(r'(Deles)([ ,;!?.])', r'De eles\2', text, flags=re.U)
    text = re.sub(r'(Delas)([ ,;!?.])', r'De elas\2', text, flags=re.U)

    text = re.sub(r'( )(deste)([ ,;!?.])', r'\1de este\3', text, flags=re.U)
    text = re.sub(r'( )(desta)([ ,;!?.])', r'\1de esta\3', text, flags=re.U)
    text = re.sub(r'( )(destes)([ ,;!?.])', r'\1de estes\3', text, flags=re.U)
    text = re.sub(r'( )(destas)([ ,;!?.])', r'\1de estas\3', text, flags=re.U)
    text = re.sub(r'( )(disto)([ ,;!?.])', r'\1de isto\3', text, flags=re.U)
    text = re.sub(r'( )(desse)([ ,;!?.])', r'\1de esse\3', text, flags=re.U)
    text = re.sub(r'( )(dessa)([ ,;!?.])', r'\1de essa\3', text, flags=re.U)
    text = re.sub(r'( )(desses)([ ,;!?.])', r'\1de esses\3', text, flags=re.U)
    text = re.sub(r'( )(dessas)([ ,;!?.])', r'\1de essas\3', text, flags=re.U)
    text = re.sub(r'( )(disso)([ ,;!?.])', r'\1de isso\3', text, flags=re.U)
    text = re.sub(r'( )(daquele)([ ,;!?.])', r'\1de aquele\3', text, flags=re.U)
    text = re.sub(r'( )(daquela)([ ,;!?.])', r'\1de aquela\3', text, flags=re.U)
    text = re.sub(r'( )(daqueles)([ ,;!?.])', r'\1de aqueles\3', text, flags=re.U)
    text = re.sub(r'( )(daquelas)([ ,;!?.])', r'\1de aquelas\3', text, flags=re.U)
    text = re.sub(r'( )(daquilo)([ ,;!?.])', r'\1de aquilo\3', text, flags=re.U)
    text = re.sub(r'( )(doutro)([ ,;!?.])', r'\1de outro\3', text, flags=re.U)
    text = re.sub(r'( )(doutra)([ ,;!?.])', r'\1de outra\3', text, flags=re.U)
    text = re.sub(r'( )(doutros)([ ,;!?.])', r'\1de outros\3', text, flags=re.U)
    text = re.sub(r'( )(doutras)([ ,;!?.])', r'\1de outras\3', text, flags=re.U)
    text = re.sub(r'( )(daqui)([ ,;!?.])', r'\1de aqui\3', text, flags=re.U)
    text = re.sub(r'( )(daí)([ ,;!?.])', r'\1de aí\3', text, flags=re.U)
    text = re.sub(r'( )(dali)([ ,;!?.])', r'\1de ali\3', text, flags=re.U)
    text = re.sub(r'( )(dalém)([ ,;!?.])', r'\1de além\3', text, flags=re.U)

    text = re.sub(r'(Deste)([ ,;!?.])', r'De este\2', text, flags=re.U)
    text = re.sub(r'(Desta)([ ,;!?.])', r'De esta\2', text, flags=re.U)
    text = re.sub(r'(Destes)([ ,;!?.])', r'De estes\2', text, flags=re.U)
    text = re.sub(r'(Destas)([ ,;!?.])', r'De estas\2', text, flags=re.U)
    text = re.sub(r'(Disto)([ ,;!?.])', r'De isto\2', text, flags=re.U)
    text = re.sub(r'(Desse)([ ,;!?.])', r'De esse\2', text, flags=re.U)
    text = re.sub(r'(Dessa)([ ,;!?.])', r'De essa\2', text, flags=re.U)
    text = re.sub(r'(Desses)([ ,;!?.])', r'De esses\2', text, flags=re.U)
    text = re.sub(r'(Dessas)([ ,;!?.])', r'De essas\2', text, flags=re.U)
    text = re.sub(r'(Disso)([ ,;!?.])', r'De isso\2', text, flags=re.U)
    text = re.sub(r'(Daquele)([ ,;!?.])', r'De aquele\2', text, flags=re.U)
    text = re.sub(r'(Daquela)([ ,;!?.])', r'De aquela\2', text, flags=re.U)
    text = re.sub(r'(Daqueles)([ ,;!?.])', r'De aqueles\2', text, flags=re.U)
    text = re.sub(r'(Daquelas)([ ,;!?.])', r'De aquelas\2', text, flags=re.U)
    text = re.sub(r'(Daquilo)([ ,;!?.])', r'De aquilo\2', text, flags=re.U)
    text = re.sub(r'(Doutro)([ ,;!?.])', r'De outro\2', text, flags=re.U)
    text = re.sub(r'(Doutra)([ ,;!?.])', r'De outra\2', text, flags=re.U)
    text = re.sub(r'(Doutros)([ ,;!?.])', r'De outros\2', text, flags=re.U)
    text = re.sub(r'(Doutras)([ ,;!?.])', r'De outras\2', text, flags=re.U)
    text = re.sub(r'(Daqui)([ ,;!?.])', r'De aqui\2', text, flags=re.U)
    text = re.sub(r'(Daí)([ ,;!?.])', r'De aí\2', text, flags=re.U)
    text = re.sub(r'(Dali)([ ,;!?.])', r'De ali\2', text, flags=re.U)
    text = re.sub(r'(Dalém)([ ,;!?.])', r'De além\2', text, flags=re.U)

    text = re.sub(r'( )(no)([ ,;!?.])', r'\1em o\3', text, flags=re.U)
    text = re.sub(r'( )(na)([ ,;!?.])', r'\1em a\3', text, flags=re.U)
    text = re.sub(r'( )(nos)([ ,;!?.])', r'\1em os\3', text, flags=re.U)
    text = re.sub(r'( )(nas)([ ,;!?.])', r'\1em as\3', text, flags=re.U)

    text = re.sub(r'(No)([ ,;!?.])', r'Em o\2', text, flags=re.U)
    text = re.sub(r'(Na)([ ,;!?.])', r'Em a\2', text, flags=re.U)
    text = re.sub(r'(Nos)([ ,;!?.])', r'Em os\2', text, flags=re.U)
    text = re.sub(r'(Nas)([ ,;!?.])', r'Em as\2', text, flags=re.U)

    text = re.sub(r'( )(num)([ ,;!?.])', r'\1em um\3', text, flags=re.U)
    text = re.sub(r'( )(numa)([ ,;!?.])', r'\1em uma\3', text, flags=re.U)
    text = re.sub(r'( )(nuns)([ ,;!?.])', r'\1em uns\3', text, flags=re.U)
    text = re.sub(r'( )(numas)([ ,;!?.])', r'\1em umas\3', text, flags=re.U)

    text = re.sub(r'(Num)([ ,;!?.])', r'Em um\2', text, flags=re.U)
    text = re.sub(r'(Numa)([ ,;!?.])', r'Em uma\2', text, flags=re.U)
    text = re.sub(r'(Nuns)([ ,;!?.])', r'Em uns\2', text, flags=re.U)
    text = re.sub(r'(Numas)([ ,;!?.])', r'Em umas\2', text, flags=re.U)

    text = re.sub(r'( )(nele)([ ,;!?.])', r'\1em ele\3', text, flags=re.U)
    text = re.sub(r'( )(nela)([ ,;!?.])', r'\1em ela\3', text, flags=re.U)
    text = re.sub(r'( )(neles)([ ,;!?.])', r'\1em eles\3', text, flags=re.U)
    text = re.sub(r'( )(nelas)([ ,;!?.])', r'\1em elas\3', text, flags=re.U)

    text = re.sub(r'(Nele)([ ,;!?.])', r'Em ele\2', text, flags=re.U)
    text = re.sub(r'(Nela)([ ,;!?.])', r'Em ela\2', text, flags=re.U)
    text = re.sub(r'(Neles)([ ,;!?.])', r'Em eles\2', text, flags=re.U)
    text = re.sub(r'(Nelas)([ ,;!?.])', r'Em elas\2', text, flags=re.U)

    text = re.sub(r'( )(neste)([ ,;!?.])', r'\1em este\3', text, flags=re.U)
    text = re.sub(r'( )(nesta)([ ,;!?.])', r'\1em esta\3', text, flags=re.U)
    text = re.sub(r'( )(nestes)([ ,;!?.])', r'\1em estes\3', text, flags=re.U)
    text = re.sub(r'( )(nestas)([ ,;!?.])', r'\1em estas\3', text, flags=re.U)
    text = re.sub(r'( )(nisto)([ ,;!?.])', r'\1em isto\3', text, flags=re.U)
    text = re.sub(r'( )(nesse)([ ,;!?.])', r'\1em esse\3', text, flags=re.U)
    text = re.sub(r'( )(nessa)([ ,;!?.])', r'\1em essa\3', text, flags=re.U)
    text = re.sub(r'( )(nesses)([ ,;!?.])', r'\1em esses\3', text, flags=re.U)
    text = re.sub(r'( )(nessas)([ ,;!?.])', r'\1em essas\3', text, flags=re.U)
    text = re.sub(r'( )(nisso)([ ,;!?.])', r'\1em isso\3', text, flags=re.U)
    text = re.sub(r'( )(naquele)([ ,;!?.])', r'\1em aquele\3', text, flags=re.U)
    text = re.sub(r'( )(naquela)([ ,;!?.])', r'\1em aquela\3', text, flags=re.U)
    text = re.sub(r'( )(naqueles)([ ,;!?.])', r'\1em aqueles\3', text, flags=re.U)
    text = re.sub(r'( )(naquelas)([ ,;!?.])', r'\1em aquelas\3', text, flags=re.U)
    text = re.sub(r'( )(naquilo)([ ,;!?.])', r'\1em aquilo\3', text, flags=re.U)

    text = re.sub(r'(Neste)([ ,;!?.])', r'Em este\2', text, flags=re.U)
    text = re.sub(r'(Nesta)([ ,;!?.])', r'Em esta\2', text, flags=re.U)
    text = re.sub(r'(Nestes)([ ,;!?.])', r'Em estes\2', text, flags=re.U)
    text = re.sub(r'(Nestas)([ ,;!?.])', r'Em estas\2', text, flags=re.U)
    text = re.sub(r'(Nisto)([ ,;!?.])', r'Em isto\2', text, flags=re.U)
    text = re.sub(r'(Nesse)([ ,;!?.])', r'Em esse\2', text, flags=re.U)
    text = re.sub(r'(Nessa)([ ,;!?.])', r'Em essa\2', text, flags=re.U)
    text = re.sub(r'(Nesses)([ ,;!?.])', r'Em esses\2', text, flags=re.U)
    text = re.sub(r'(Nessas)([ ,;!?.])', r'Em essas\2', text, flags=re.U)
    text = re.sub(r'(Nisso)([ ,;!?.])', r'Em isso\2', text, flags=re.U)
    text = re.sub(r'(Naquele)([ ,;!?.])', r'Em aquele\2', text, flags=re.U)
    text = re.sub(r'(Naquela)([ ,;!?.])', r'Em aquela\2', text, flags=re.U)
    text = re.sub(r'(Naqueles)([ ,;!?.])', r'Em aqueles\2', text, flags=re.U)
    text = re.sub(r'(Naquelas)([ ,;!?.])', r'Em aquelas\2', text, flags=re.U)
    text = re.sub(r'(Naquilo)([ ,;!?.])', r'Em aquilo\2', text, flags=re.U)

    text = re.sub(r'( )(pelo)([ ,;!?.])', r'\1por o\3', text, flags=re.U)
    text = re.sub(r'( )(pela)([ ,;!?.])', r'\1por a\3', text, flags=re.U)
    text = re.sub(r'( )(pelos)([ ,;!?.])', r'\1por os\3', text, flags=re.U)
    text = re.sub(r'( )(pelas)([ ,;!?.])', r'\1por as\3', text, flags=re.U)

    text = re.sub(r'(Pelo)([ ,;!?.])', r'Por o\2', text, flags=re.U)
    text = re.sub(r'(Pela)([ ,;!?.])', r'Por a\2', text, flags=re.U)
    text = re.sub(r'(Pelos)([ ,;!?.])', r'Por os\2', text, flags=re.U)
    text = re.sub(r'(Pelas)([ ,;!?.])', r'Por as\2', text, flags=re.U)

    text = text.replace(u' à ', u' a a ')
    text = text.replace(u'À ', u'A a ')
    text = re.sub(r'( )(ao)([ ,;!?.])', r'\1a o\3', text, flags=re.U)
    text = re.sub(r'(Ao)([ ,;!?.])', r'A o\2', text, flags=re.U)

    return text

def realize(text):
    text = re.sub(r'( )(com um)([ ,;!?.])', r'\1 cum\3', text, flags=re.U)
    text = re.sub(r'(Com um)([ ,;!?.])', r'Cum\2', text, flags=re.U)

    text = re.sub(r'( )(de o)([ ,;!?.])', r'\1do\3', text, flags=re.U)
    text = re.sub(r'( )(de a)([ ,;!?.])', r'\1da\3', text, flags=re.U)
    text = re.sub(r'( )(de os)([ ,;!?.])', r'\1dos\3', text, flags=re.U)
    text = re.sub(r'( )(de as)([ ,;!?.])', r'\1das\3', text, flags=re.U)

    text = re.sub(r'(De o)([ ,;!?.])', r'Do\2', text, flags=re.U)
    text = re.sub(r'(De a)([ ,;!?.])', r'Da\2', text, flags=re.U)
    text = re.sub(r'(De os)([ ,;!?.])', r'Dos\2', text, flags=re.U)
    text = re.sub(r'(De as)([ ,;!?.])', r'Das\2', text, flags=re.U)

    text = re.sub(r'( )(de um)([ ,;!?.])', r'\1dum\3', text, flags=re.U)
    text = re.sub(r'( )(de uma)([ ,;!?.])', r'\1duma\3', text, flags=re.U)
    text = re.sub(r'( )(de uns)([ ,;!?.])', r'\1duns\3', text, flags=re.U)
    text = re.sub(r'( )(de umas)([ ,;!?.])', r'\1dumas\3', text, flags=re.U)

    text = re.sub(r'(De um)([ ,;!?.])', r'Dum\2', text, flags=re.U)
    text = re.sub(r'(De uma)([ ,;!?.])', r'Duma\2', text, flags=re.U)
    text = re.sub(r'(De uns)([ ,;!?.])', r'Duns\2', text, flags=re.U)
    text = re.sub(r'(De umas)([ ,;!?.])', r'Dumas\2', text, flags=re.U)

    text = re.sub(r'( )(de ele)([ ,;!?.])', r'\1dele\3', text, flags=re.U)
    text = re.sub(r'( )(de ela)([ ,;!?.])', r'\1dela\3', text, flags=re.U)
    text = re.sub(r'( )(de eles)([ ,;!?.])', r'\1deles\3', text, flags=re.U)
    text = re.sub(r'( )(de elas)([ ,;!?.])', r'\1delas\3', text, flags=re.U)

    text = re.sub(r'(De ele)([ ,;!?.])', r'Dele\2', text, flags=re.U)
    text = re.sub(r'(De ela)([ ,;!?.])', r'Dela\2', text, flags=re.U)
    text = re.sub(r'(De eles)([ ,;!?.])', r'Deles\2', text, flags=re.U)
    text = re.sub(r'(De elas)([ ,;!?.])', r'Delas\2', text, flags=re.U)

    text = re.sub(r'( )(de este)([ ,;!?.])', r'\1deste\3', text, flags=re.U)
    text = re.sub(r'( )(de esta)([ ,;!?.])', r'\1desta\3', text, flags=re.U)
    text = re.sub(r'( )(de estes)([ ,;!?.])', r'\1destes\3', text, flags=re.U)
    text = re.sub(r'( )(de estas)([ ,;!?.])', r'\1destas\3', text, flags=re.U)
    text = re.sub(r'( )(de isto)([ ,;!?.])', r'\1disto\3', text, flags=re.U)
    text = re.sub(r'( )(de esse)([ ,;!?.])', r'\1desse\3', text, flags=re.U)
    text = re.sub(r'( )(de essa)([ ,;!?.])', r'\1dessa\3', text, flags=re.U)
    text = re.sub(r'( )(de esses)([ ,;!?.])', r'\1desses\3', text, flags=re.U)
    text = re.sub(r'( )(de essas)([ ,;!?.])', r'\1dessas\3', text, flags=re.U)
    text = re.sub(r'( )(de isso)([ ,;!?.])', r'\1disso\3', text, flags=re.U)
    text = re.sub(r'( )(de aquele)([ ,;!?.])', r'\1daquele\3', text, flags=re.U)
    text = re.sub(r'( )(de aquela)([ ,;!?.])', r'\1daquela\3', text, flags=re.U)
    text = re.sub(r'( )(de aqueles)([ ,;!?.])', r'\1daqueles\3', text, flags=re.U)
    text = re.sub(r'( )(de aquelas)([ ,;!?.])', r'\1daquelas\3', text, flags=re.U)
    text = re.sub(r'( )(de aquilo)([ ,;!?.])', r'\1daquilo\3', text, flags=re.U)
    text = re.sub(r'( )(de outro)([ ,;!?.])', r'\1doutro\3', text, flags=re.U)
    text = re.sub(r'( )(de outra)([ ,;!?.])', r'\1doutra\3', text, flags=re.U)
    text = re.sub(r'( )(de outros)([ ,;!?.])', r'\1doutros\3', text, flags=re.U)
    text = re.sub(r'( )(de outras)([ ,;!?.])', r'\1doutras\3', text, flags=re.U)
    text = re.sub(r'( )(de aqui)([ ,;!?.])', r'\1daqui\3', text, flags=re.U)
    text = re.sub(r'( )(de aí)([ ,;!?.])', r'\1daí\3', text, flags=re.U)
    text = re.sub(r'( )(de ali)([ ,;!?.])', r'\1dali\3', text, flags=re.U)
    text = re.sub(r'( )(de além)([ ,;!?.])', r'\1dalém\3', text, flags=re.U)

    text = re.sub(r'(De este)([ ,;!?.])', r'Deste\2', text, flags=re.U)
    text = re.sub(r'(De esta)([ ,;!?.])', r'Desta\2', text, flags=re.U)
    text = re.sub(r'(De estes)([ ,;!?.])', r'Destes\2', text, flags=re.U)
    text = re.sub(r'(De estas)([ ,;!?.])', r'Destas\2', text, flags=re.U)
    text = re.sub(r'(De isto)([ ,;!?.])', r'Disto\2', text, flags=re.U)
    text = re.sub(r'(De esse)([ ,;!?.])', r'Desse\2', text, flags=re.U)
    text = re.sub(r'(De essa)([ ,;!?.])', r'Dessa\2', text, flags=re.U)
    text = re.sub(r'(De esses)([ ,;!?.])', r'Desses\2', text, flags=re.U)
    text = re.sub(r'(De essas)([ ,;!?.])', r'Dessas\2', text, flags=re.U)
    text = re.sub(r'(De isso)([ ,;!?.])', r'Disso\2', text, flags=re.U)
    text = re.sub(r'(De aquele)([ ,;!?.])', r'Daquele\2', text, flags=re.U)
    text = re.sub(r'(De aquela)([ ,;!?.])', r'Daquela\2', text, flags=re.U)
    text = re.sub(r'(De aqueles)([ ,;!?.])', r'Daqueles\2', text, flags=re.U)
    text = re.sub(r'(De aquelas)([ ,;!?.])', r'Daquelas\2', text, flags=re.U)
    text = re.sub(r'(De aquilo)([ ,;!?.])', r'Daquilo\2', text, flags=re.U)
    text = re.sub(r'(De outro)([ ,;!?.])', r'Doutro\2', text, flags=re.U)
    text = re.sub(r'(De outra)([ ,;!?.])', r'Doutra\2', text, flags=re.U)
    text = re.sub(r'(De outros)([ ,;!?.])', r'Doutros\2', text, flags=re.U)
    text = re.sub(r'(De outras)([ ,;!?.])', r'Doutras\2', text, flags=re.U)
    text = re.sub(r'(De aqui)([ ,;!?.])', r'Daqui\2', text, flags=re.U)
    text = re.sub(r'(De aí)([ ,;!?.])', r'Daí\2', text, flags=re.U)
    text = re.sub(r'(De ali)([ ,;!?.])', r'Dali\2', text, flags=re.U)
    text = re.sub(r'(De além)([ ,;!?.])', r'Dalém\2', text, flags=re.U)

    text = re.sub(r'( )(em o)([ ,;!?.])', r'\1no\3', text, flags=re.U)
    text = re.sub(r'( )(em a)([ ,;!?.])', r'\1na\3', text, flags=re.U)
    text = re.sub(r'( )(em os)([ ,;!?.])', r'\1nos\3', text, flags=re.U)
    text = re.sub(r'( )(em as)([ ,;!?.])', r'\1nas\3', text, flags=re.U)

    text = re.sub(r'(Em o)([ ,;!?.])', r'No\2', text, flags=re.U)
    text = re.sub(r'(Em a)([ ,;!?.])', r'Na\2', text, flags=re.U)
    text = re.sub(r'(Em os)([ ,;!?.])', r'Nos\2', text, flags=re.U)
    text = re.sub(r'(Em as)([ ,;!?.])', r'Nas\2', text, flags=re.U)

    text = re.sub(r'( )(em um)([ ,;!?.])', r'\1num\3', text, flags=re.U)
    text = re.sub(r'( )(em uma)([ ,;!?.])', r'\1numa\3', text, flags=re.U)
    text = re.sub(r'( )(em uns)([ ,;!?.])', r'\1nuns\3', text, flags=re.U)
    text = re.sub(r'( )(em umas)([ ,;!?.])', r'\1numas\3', text, flags=re.U)

    text = re.sub(r'(eEm um)([ ,;!?.])', r'Num\2', text, flags=re.U)
    text = re.sub(r'(Em uma)([ ,;!?.])', r'Numa\2', text, flags=re.U)
    text = re.sub(r'(Em uns)([ ,;!?.])', r'Nuns\2', text, flags=re.U)
    text = re.sub(r'(Em umas)([ ,;!?.])', r'Numas\2', text, flags=re.U)

    text = re.sub(r'( )(em ele)([ ,;!?.])', r'\1nele\3', text, flags=re.U)
    text = re.sub(r'( )(em ela)([ ,;!?.])', r'\1nela\3', text, flags=re.U)
    text = re.sub(r'( )(em eles)([ ,;!?.])', r'\1neles\3', text, flags=re.U)
    text = re.sub(r'( )(em elas)([ ,;!?.])', r'\1nelas\3', text, flags=re.U)

    text = re.sub(r'(Em ele)([ ,;!?.])', r'Nele\2', text, flags=re.U)
    text = re.sub(r'(Em ela)([ ,;!?.])', r'Nela\2', text, flags=re.U)
    text = re.sub(r'(Em eles)([ ,;!?.])', r'Neles\2', text, flags=re.U)
    text = re.sub(r'(Em elas)([ ,;!?.])', r'Nelas\2', text, flags=re.U)

    text = re.sub(r'( )(em este)([ ,;!?.])', r'\1neste\3', text, flags=re.U)
    text = re.sub(r'( )(em esta)([ ,;!?.])', r'\1nesta\3', text, flags=re.U)
    text = re.sub(r'( )(em estes)([ ,;!?.])', r'\1nestes\3', text, flags=re.U)
    text = re.sub(r'( )(em estas)([ ,;!?.])', r'\1nestas\3', text, flags=re.U)
    text = re.sub(r'( )(em isto)([ ,;!?.])', r'\1nisto\3', text, flags=re.U)
    text = re.sub(r'( )(em esse)([ ,;!?.])', r'\1nesse\3', text, flags=re.U)
    text = re.sub(r'( )(em essa)([ ,;!?.])', r'\1nessa\3', text, flags=re.U)
    text = re.sub(r'( )(em esses)([ ,;!?.])', r'\1nesses\3', text, flags=re.U)
    text = re.sub(r'( )(em essas)([ ,;!?.])', r'\1nessas\3', text, flags=re.U)
    text = re.sub(r'( )(em isso)([ ,;!?.])', r'\1nisso\3', text, flags=re.U)
    text = re.sub(r'( )(em aquele)([ ,;!?.])', r'\1naquele\3', text, flags=re.U)
    text = re.sub(r'( )(em aquela)([ ,;!?.])', r'\1naquela\3', text, flags=re.U)
    text = re.sub(r'( )(em aqueles)([ ,;!?.])', r'\1naqueles\3', text, flags=re.U)
    text = re.sub(r'( )(em aquelas)([ ,;!?.])', r'\1naquelas\3', text, flags=re.U)
    text = re.sub(r'( )(em aquilo)([ ,;!?.])', r'\1naquilo\3', text, flags=re.U)

    text = re.sub(r'(Em este)([ ,;!?.])', r'Neste\2', text, flags=re.U)
    text = re.sub(r'(Em esta)([ ,;!?.])', r'Nesta\2', text, flags=re.U)
    text = re.sub(r'(Em estes)([ ,;!?.])', r'Nestes\2', text, flags=re.U)
    text = re.sub(r'(Em estas)([ ,;!?.])', r'Nestas\2', text, flags=re.U)
    text = re.sub(r'(Em isto)([ ,;!?.])', r'Nisto\2', text, flags=re.U)
    text = re.sub(r'(Em esse)([ ,;!?.])', r'Nesse\2', text, flags=re.U)
    text = re.sub(r'(Em essa)([ ,;!?.])', r'Nessa\2', text, flags=re.U)
    text = re.sub(r'(Em esses)([ ,;!?.])', r'Nesses\2', text, flags=re.U)
    text = re.sub(r'(Em essas)([ ,;!?.])', r'Nessas\2', text, flags=re.U)
    text = re.sub(r'(Em isso)([ ,;!?.])', r'Nisso\2', text, flags=re.U)
    text = re.sub(r'(Em aquele)([ ,;!?.])', r'Naquele\2', text, flags=re.U)
    text = re.sub(r'(Em aquela)([ ,;!?.])', r'Naquela\2', text, flags=re.U)
    text = re.sub(r'(Em aqueles)([ ,;!?.])', r'Naqueles\2', text, flags=re.U)
    text = re.sub(r'(Em aquelas)([ ,;!?.])', r'Naquelas\2', text, flags=re.U)
    text = re.sub(r'(Em aquilo)([ ,;!?.])', r'Naquilo\2', text, flags=re.U)

    text = re.sub(r'( )(por o)([ ,;!?.])', r'\1pelo\3', text, flags=re.U)
    text = re.sub(r'( )(por a)([ ,;!?.])', r'\1pela\3', text, flags=re.U)
    text = re.sub(r'( )(por os)([ ,;!?.])', r'\1pelos\3', text, flags=re.U)
    text = re.sub(r'( )(por as)([ ,;!?.])', r'\1pelas\3', text, flags=re.U)

    text = re.sub(r'(Por o)([ ,;!?.])', r'Pelo\2', text, flags=re.U)
    text = re.sub(r'(Por a)([ ,;!?.])', r'Pela\2', text, flags=re.U)
    text = re.sub(r'(Por os)([ ,;!?.])', r'Pelos\2', text, flags=re.U)
    text = re.sub(r'(Por as)([ ,;!?.])', r'Pelas\2', text, flags=re.U)

    text = text.replace(u' a a ', u' à ')
    text = text.replace(u'A a ', u'À ')
    text = text.replace(u' a as ', u' às ')
    text = text.replace(u'A as ', u'Às ')

    text = re.sub(r'( )(a o)([ ,;!?.])', r'\1ao\3', text, flags=re.U)
    text = re.sub(r'(A o)([ ,;!?.])', r'Ao\2', text, flags=re.U)
    text = re.sub(r'( )(a os)([ ,;!?.])', r'\1aos\3', text, flags=re.U)
    text = re.sub(r'(A os)([ ,;!?.])', r'Aos\2', text, flags=re.U)

    return text

if __name__ == '__main__':
    fread = 'data2019/tok/pt_bosque-Pred-Stanford.conllu'
    fout = 'data2019/tok/pt_bosque-Pred-Stanford.conllu'

    with open(fread) as f:
        doc = f.read().decode('utf-8')
    doc = realize(doc)

    with open(fout, 'w') as f:
        f.write(doc.encode('utf-8'))
        
    fread = 'data2019/detok/pt_bosque-Pred-Stanford.conllu'
    fout = 'data2019/detok/pt_bosque-Pred-Stanford.conllu'

    with open(fread) as f:
        doc = f.read().decode('utf-8')
    doc = realize(doc)

    with open(fout, 'w') as f:
        f.write(doc.encode('utf-8'))
        
    ###########################################
    fread = 'data2019/tok/pt_bosque-ud-test.conllu'
    fout = 'data2019/tok/pt_bosque-ud-test.conllu'

    with open(fread) as f:
        doc = f.read().decode('utf-8')
    doc = realize(doc)

    with open(fout, 'w') as f:
        f.write(doc.encode('utf-8'))
        
    fread = 'data2019/detok/pt_bosque-ud-test.conllu'
    fout = 'data2019/detok/pt_bosque-ud-test.conllu'

    with open(fread) as f:
        doc = f.read().decode('utf-8')
    doc = realize(doc)

    with open(fout, 'w') as f:
        f.write(doc.encode('utf-8'))
        
    ###########################################
    fread = 'data2019/tok/pt_gsd-ud-test.conllu'
    fout = 'data2019/tok/pt_gsd-ud-test.conllu'

    with open(fread) as f:
        doc = f.read().decode('utf-8')
    doc = realize(doc)

    with open(fout, 'w') as f:
        f.write(doc.encode('utf-8'))
        
    fread = 'data2019/detok/pt_gsd-ud-test.conllu'
    fout = 'data2019/detok/pt_gsd-ud-test.conllu'

    with open(fread) as f:
        doc = f.read().decode('utf-8')
    doc = realize(doc)

    with open(fout, 'w') as f:
        f.write(doc.encode('utf-8'))
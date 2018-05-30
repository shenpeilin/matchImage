import utils
import codecs,json
import numpy

class imgMatch:
    def __init__(self):
        self.template = utils.load_model('./models/MANO_RIGHT.pkl')
        print self.template
        # self.J = self.template['J_regressor'].dot(self.template['v_template'])
        # with open( 'J.off', 'w') as fp:
        #     fp.write('COFF\n')
        #     fp.write('%d 0 0\n' % (len(self.J)))
        #     for nodeIdx in range(0, len(self.J)):
        #         v = self.J[nodeIdx]
        #         fp.write('%f %f %f 255 0 0 255\n' % (v[0], v[1], v[2]))

        # print 'nodeMesh saved to J.off'
    
    def jsonSave(self):
        temObj = {}
        temObj['hands_components'] = self.template['hands_components'].tolist()
        temObj['f'] = self.template['f'].tolist()
        temObj['J_regressor'] = self.template['J_regressor'].toarray().tolist()
        temObj['kintree_table'] = self.template['kintree_table'].tolist()
        temObj['hands_coeffs'] = self.template['hands_coeffs'].tolist()
        temObj['weights'] = self.template['weights'].tolist()
        temObj['posedirs'] = self.template['posedirs'].tolist()
        temObj['hands_mean'] = self.template['hands_mean'].tolist()
        temObj['v_template'] = self.template['v_template'].tolist()
        temObj['shapedirs'] = numpy.array(self.template['shapedirs']).tolist()
        jsonObj = json.dump(temObj, codecs.open('model.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)

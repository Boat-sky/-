import datetime
import logging
logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(message)s')

logmsg = ''
timemark = dict()
saveDefault = False
def log(msg, save=None, oneline=False):
	global logmsg
	global saveDefault
	time = datetime.datetime.now()
	tem = '%s: %s' % (time, msg)
	if save != None:
		if save:
			logmsg += tem + '\n'
	elif saveDefault:
		logmsg += tem + '\n'
	if oneline:
		#print(tem, end='\r')
		#logging.info('\r' + tem)
		logging.info(tem)
	else:
		#print(tem)
		logging.info(tem)


def marktime(marker):
	global timemark
	timemark[marker] = datetime.datetime.now()


if __name__ == '__main__':
	log('')
from config import MONGODB_HOST, MONGODB_PORT, MONGODB_NAME, MONGODB_USERNAME, MONGODB_PASSWORD

with open('./fireworks_files/my_launchpad.yaml', 'w') as f:
	f.write('host: %s\n' % MONGODB_HOST)
	f.write('port: %d\n' % MONGODB_PORT)
	f.write('name: %s\n' % MONGODB_NAME)
	f.write('username: %s\n' % MONGODB_USERNAME)
	f.write('password: %s\n' % MONGODB_PASSWORD)
	f.write('logdir: null\n')
	f.write('strm_lvl: INFO\n')
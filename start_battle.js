const Sim = require('pokemon-showdown');
stream = new Sim.BattleStream();

(async () => {
	let noDataTimeout;
	
    for await (const output of stream) {
        console.log(output);
		
		if (noDataTimeout) {
			clearTimeout(noDataTimeout);
		}

		noDataTimeout = setTimeout(() => {
			console.log('await'); 
		}, 1);
    }
})();

process.stdin.on('data', (data) => {
    stream.write(data.toString().trim());
});
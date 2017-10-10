WAVE_TYPES <- c('SS', 'SWA', 'delta', 'theta', 'alpha', 'beta')
ylabels <- c('num. of spindles/3min', '%SWA/20sec', 'Amp/3min', 'Amp/3min', 'Amp/3min', 'Amp/3min')

args = commandArgs(TRUE)
if (length(args)==0) {
	stop("Input file must be supplied (without extension, e.g. *_<wave>_hipnogram.csv).n", call.=False)
} else if (length(args==1)) {
	name <- args[1]
}

w_id <- 1
xlabel <- ''
png(paste0(name, '.png'), width=1000, height=700)
par(mfrow = c(length(WAVE_TYPES), 1), mar=c(4, 4.5, 4, 1)) #bottom, left, top and right margins
for (wave in WAVE_TYPES) {
    file_name <- paste0(name, '_', wave, '_hipnogram.csv')
    hipnogram <- read.csv(file_name)
    occ <- hipnogram[['occurences']]
    names(occ) <- round(hipnogram[['time']], digits=1)
    if (w_id == length(WAVE_TYPES)){ xlabel='time (hours)' }
    barplot(occ, main=wave, ylab=ylabels[w_id], xlab=xlabel, cex.axis=1.5, cex.lab=1.5, cex.main=2)  #, xlim=c(0, 16*20))
	# axis(1, at=1:10, labels=letters[1:10])
	#if (wave=='SWA') {abline(h=20, col = "gray60")}
    w_id <- w_id + 1
}
dev.off()

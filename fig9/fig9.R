library(Hmisc) # rcorr
library(car) #VIF
library(psych) # describe / alpha
library(plyr) #mapvalues
library(irr)
library(cowplot) # plot_grid
#options(scipen = 999) # disable scientific notation
library(reshape2)
library("scales")

rm(list = ls())
# setwd("~/Dropbox/Research/Image clustering/R/")


raw <- data.frame(read.csv(file="img protest.txt", header=TRUE, quote = "", sep="\t", comment.char = ""))
colnames(raw)
r = raw
colnames(r)
nrow(raw); nrow(r)
r
 
dm = melt(r,id.vars=c("Dataset","X..of.clusters"))

colnames(dm) = c("dataset","nc","clusterid","perc")
dm
dm$nc = factor(dm$nc, levels = unique(dm$nc))
dm$highlight = ifelse(dm$clusterid == "Average","Y","N")
dm

ggplot(dm, aes(y=perc, x=clusterid, fill = highlight)) + 
  facet_grid(nc ~ dataset) +
  geom_bar(width=0.3, stat="identity")+   # +  # reverse the order of stacks
  theme_minimal(base_size=14) + #coord_flip() +
  scale_x_discrete(labels = c("1","2","3","4","5","6","7","8","9","10",expression(italic("M"))))+
  # scale_y_continuous(expand=c(0,0))+
  scale_fill_manual(values=c("darkgray", "firebrick3")) +
  geom_text(aes(x = clusterid, y = 1.2, label = round (perc, 2)), size = 2)+
  theme(text=element_text(size=14,family="Arial"),
        axis.ticks = element_blank(), axis.line = element_blank(),
        panel.spacing = unit(2, "lines"),
        legend.position = "none",
        #panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank()) +
  xlab("Cluster ID") + ylab("Within-cluster consistency")


w = 7.5; h = 6; imgname = paste('percent protest.png', sep = ' '); imgname
ggsave(imgname,width=w,height=h,dpi=300,bg="white")


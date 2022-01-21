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


raw <- data.frame(read.csv(file="img climate.txt", header=TRUE, quote = "", sep="\t", comment.char = ""))
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


w = 7.5; h = 6; imgname = paste('fig16.png', sep = ' '); imgname
ggsave(imgname,width=w,height=h,dpi=300,bg="white")


# ttest
library(purrr)
library(gtools)

r$N <- as.numeric(sapply(strsplit(as.character(r$`X..of.clusters`), " "), head, 1))

t.test(discard(r[1, 3:10], is.na), discard(r[5, 3:10], is.na), var.eq = T)

idx <- combinations(n = 6, r = 2, v = 1:6, repeats.allowed = FALSE)

df <- c()
r$Dataset <- as.character(r$Dataset)
r$X..of.clusters <- as.character(r$X..of.clusters)
for (i in 1:nrow(idx)){
  id1 <- idx[i, 1]
  id2 <- idx[i, 2]
  n1 <- r[id1, 14] * 30
  n2 <- r[id2, 14] * 30
  
  print (unname(r[id1, 1:2]))
  print (unname(r[id2, 1:2]))
  # print (paste (unname(r[id1, 1:2]), unname(r[id1, 1:2])));
  ss <- prop.test(c(r[id1, 13] * n1, r[id2, 13] * n2),  c(n1, n2))
  # print (prop.test(c(r[id1, 13] * n1, r[id2, 13] * n2),  c(n1, n2)))
  print (ss$p.value)
  
  df <- rbind (df, c(paste(r[id1, 1], ", K=", r[id1, 14]) ,paste(r[id2, 1],", K=",  r[id2, 14]), ss$estimate, ss$p.value ))
  print ("----------------------------------------")
}
df <- as.data.frame(df)

names(df) <- c("Algorithm1",  "Algorithm2", "Within-cluster consistency1","Within-cluster consistency2", "p-value")
df$`p-value` <- as.numeric(as.character(df$`p-value`))

## output to latex -----
library(xtable)

p.val <- df$`p-value`
stars.pval(p.val)
df$`p-value` <- paste(round(df$`p-value`, 3), stars.pval(p.val))

xtable(df)


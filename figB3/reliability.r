
library(ggplot2)

d <- read.csv("reliability.csv")


ggplot(d, aes(y=kappa, x = Dataset, fill = Algorithm)) + 
  facet_grid(~ N) +
  geom_bar(width=0.3, stat="identity",position="dodge")+   # +  # reverse the order of stacks
  theme_minimal(base_size=14) + 
  scale_fill_grey()+
  xlab ("Number of clusters") + 
  # scale_y_continuous(expand=c(0,0))+
  theme(text=element_text(size=14,family="Arial"),
        axis.ticks = element_blank(), axis.line = element_blank(),
        panel.spacing = unit(2, "lines"),
        panel.grid.minor = element_blank()) +
  ylab("Cohen's Kappa")
w = 7.5; h = 6;
#imgname = paste('percent',w,h,'.png', sep = ' '); imgname
ggsave("intercoder.png",width=w,height=h,dpi=300,bg="white")

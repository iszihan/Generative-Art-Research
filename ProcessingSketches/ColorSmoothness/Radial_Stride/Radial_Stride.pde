int radius;
int iteration;
float back_col;
float smt;

void setup() {
  size(224, 224);
  noStroke();
  ellipseMode(RADIUS);
  noLoop();
  
}

void draw() {
 
 
  //for( iteration = 1; iteration <=100; iteration ++){
    back_col = random(0,100);
    radius = int(random(50,150));
    for( int level = 0; level <= 10; level+=1){
      smt = map(level,0,10,1.0,radius/2);
      //for( int num=1; num<=3; num++){
        iteration =1;
        int num=1;
        background(back_col);
        drawGradientSet(num,smt);
       
        PImage temp = get();
        String fileName = String.format("Output/tri-c%02d-%03d%01d", level, iteration,num);
        temp.save(fileName);
    
        clear();
      }
    //}
  }
         
    

void drawGradientSet(int num, float smt){
  for(int i=1; i<= num; i++){
    float x = random(10,width-10);
    float y = random(10,height-10);
    drawGradient(x, y, smt);
    
  }

}
void drawGradient(float x, float y,float smt) {
  
  for (float r = radius; r >= 0; r-=smt) {
    
    float col =  255-((r-smt)/(radius-smt))*(255-back_col);
    float alpha;
    if(r == radius && smt == radius/2){
      alpha = 0;
    }
    else{
      alpha = (255-(r-smt)/radius*255);
    }
    fill(col,alpha);
    //fill(255-r/radius*255);
    ellipse(x, y, r, r);
  }
}

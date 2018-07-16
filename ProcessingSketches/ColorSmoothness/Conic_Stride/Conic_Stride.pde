color b1,b2;
float smt, nstep;
void setup(){
  size(224,224);
  noLoop();
}

void draw(){
  int r1 = 0;
  int r2 = 255;
  b1 = color(r1);
  b2 = color(r2);
  for(int level=0; level<10;level++){
    
    if(level == 0){
     nstep = width/2;
    }
    else if(level <= 5){
     nstep = map(level,1,5,10,5);
    }
    else if(level>= 6){
     nstep = map(level,6,9,4,1);
    }
   
    smt = width/nstep;
    conicGradient(r1,r2,smt);
    
    PImage temp = get();
    String fileName = String.format("Output/tri-c%02d.png",level);
    temp.save(fileName);
    
    
  }
  
 
 
}

void conicGradient(int r1, int r2, float smt){
  
  
  for(int x = 0; x<= width-smt; x+=smt){
    
    float inter = map(x,0,4*width-smt,0,1);
    color col = lerpColor(r1,r2,inter);
    stroke(col);
    fill(col);
    //line(width/2,height/2,x,0);
    triangle(x,0,x+smt,0,width/2,height/2);
   
  }
  
  for(int y = 0; y<= height-smt; y+=smt){
    
    float inter = map(y+width,0,4*width-smt,0,1);
    color col = lerpColor(r1,r2,inter);
    stroke(col);
    fill(col);
    //line(width/2,height/2,x,0);
    triangle(width,y,width,y+smt,width/2,height/2);
  }
  
  for(int x = 0; x<= width-smt; x+=smt){
    
    float inter = map(x+2*width,0,4*width-smt,0,1);
    color col = lerpColor(r1,r2,inter);
    stroke(col);
    fill(col);
    triangle(width-x,height,(width-x)-smt,height,width/2,height/2);
  }
  
  for(int y = 0; y<= height-smt; y+=smt){
    
    float inter = map(y+3*width,0,4*width-smt,0,1);
    color col = lerpColor(r1,r2,inter);
    stroke(col);
    fill(col);
    //line(width/2,height/2,x,0);
    triangle(0,height-y,0,(height-y)-smt,width/2,height/2);
  } 
  
}

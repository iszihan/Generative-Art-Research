PGraphics mask_vert;

void setup() {

  size(224, 224);
  noLoop();
  ellipseMode(RADIUS);
}


void draw() {


  //for (int incrt = 1000; incrt >= 0; incrt-=100) {
  for (int y = 0; y<= height; y+=20) {
    int incrt=0;

    float r = dist(0, height/2, width/2, height/2-incrt);
    noFill();
    stroke(0);
    arc(width/2, height+incrt-y, r, r, PI, TWO_PI);
    //arc(width/2, height/2+incrt-y, r, r, PI, TWO_PI);
  }

  PImage temp = get();

  mask_vert = createGraphics(width, height);
  mask_vert.beginDraw();
  mask_vert.rect(0, 0, width, height/2);
  mask_vert.endDraw(); 
  imageFlipped_vert(temp,0,0);
  temp.mask(mask_vert);
  image(temp,0,0);
  
  //String filename = String.format("Output/tri-r%02d-cv.png", incrt/100);
  //temp.save(filename);
  //}
}


void imageFlipped_vert( PImage img, float x, float y ){
  pushMatrix(); 
  translate(0, img.height);
  scale( 1, -1 );
  image(img, x, y); 
  popMatrix(); 
} 

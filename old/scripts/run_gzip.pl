#!/usr/bin/perl

use strict;
use warnings;
use Time::HiRes;
use Math::BigFloat;
Math::BigFloat->precision(-3);
BEGIN {push @INC, '/fs/buzz/ukoc'}
use util qw/harvestLCovFile/;

my $SUT_VAN="gzip-1.6-vanilla";
my $SUT_NAME="gzip-1.6";

my $SUT_DIR="/fs/buzz/ukoc/gzip";
my $SRC_DIR="$SUT_DIR/$SUT_NAME";
my $ARC_DIR="$SUT_DIR/archive";
my $BUILD_DIR="$SRC_DIR/build";
my $TEST_DIR="$SUT_DIR/testfiles";

my $config=$ARGV[0];
my $confSpaceFile="$SUT_DIR/gzip.dom";
sub prepare($){
    my $conf=shift;
   `cd $SUT_DIR && rm -fr $SUT_NAME $TEST_DIR`;
   `cd $SUT_DIR && cp -R $SUT_VAN $SUT_NAME`;
   `mkdir $BUILD_DIR`;
   `cp -R $TEST_DIR-vanilla $TEST_DIR`;
   `cd $ARC_DIR && rm -fr test.*.log build.*.log`;
   
   my $result="";
   my @options=split(" , ", $conf);
   foreach my $o (@options){
       my @pair=split(" ", $o);
       if($pair[1] eq "on"){ $result=$result." ".$pair[0];}
   }
   return $result;
}

sub buildSUT(){
   # Configure
   my $start = Time::HiRes::time();
   my $logFile = "$ARC_DIR/build.$start.log";
   `echo "---Configuration:$config,Start_Time:$start\n" > $logFile 2>&1`;
   my $cov="CFLAGS='--coverage' CCFLAGS='--coverage' CCPFLAGS='--coverage'";
   `cd $SRC_DIR && ./configure --prefix=$BUILD_DIR $cov >> $logFile 2>&1`;
   `echo "\n---Configuration Completed, Continuing with make!!!\n" >> $logFile 2>&1`;
   
   # Make
   `cd $SRC_DIR && make >> $logFile 2>&1`;
   `echo "\n---Make Completed, Continuing with make install!!!\n" >> $logFile 2>&1`;
   
   # Make install
   `cd $SRC_DIR && make install >> $logFile 2>&1`;
   my $time = Math::BigFloat->new(Time::HiRes::time() - $start);
   `echo "\n---Installation Completed, Total Time:$time---\n" >> $logFile 2>&1`;
   
   return $start;
}

# !!! runTest subroutine should be customized for SUT !!!
sub runTests($$){
   my $config = shift;
   my $time = shift;
   my $testLogFile="$ARC_DIR/test.$time.log";
   `echo "\n---Testing Starts with configuration:$config ---\n" > $testLogFile 2>&1`;
   # run test
   #`cd $BUILD_DIR/bin && ./gzip $config >> $testLogFile 2>&1`;
   `cd $BUILD_DIR/bin && ./gzip $config $TEST_DIR/textfile.txt >> $testLogFile 2>&1`;
   `cd $BUILD_DIR/bin && ./gzip $config $TEST_DIR/pdffile.pdf >> $testLogFile 2>&1`;
   `cd $BUILD_DIR/bin && ./gzip $config $TEST_DIR/picture.jpg >> $testLogFile 2>&1`;
   `cd $BUILD_DIR/bin && ./gzip $config $TEST_DIR/testfolder >> $testLogFile 2>&1`;
   `cd $BUILD_DIR/bin && ./gzip $config $TEST_DIR/tarredfolder.tar >> $testLogFile 2>&1`;
   `cd $BUILD_DIR/bin && ./gzip $config $TEST_DIR/picture2.jpg.gz >> $testLogFile 2>&1`;
   `cd $BUILD_DIR/bin && ./gzip $config $TEST_DIR/tarredfolder2.tar.gz >> $testLogFile 2>&1`;
   `cd $BUILD_DIR/bin && ./gzip $config $TEST_DIR/audiofile.mp3 >> $testLogFile 2>&1`;
   `cd $BUILD_DIR/bin && ./gzip $config $TEST_DIR/moviefile.mp4 >> $testLogFile 2>&1`;
   
   #Harvesting coverage information
   my $covFile = "$ARC_DIR/cov.$time.txt";
   my $aggregatedCovFile = "$SUT_DIR/aggregated.gzip.cov";
   `cd $BUILD_DIR/bin && lcov -c --ignore-errors='gcov,source' -d $SRC_DIR -o $aggregatedCovFile > cov.log 2>&1`;
   util::harvestLCovFile($aggregatedCovFile, $confSpaceFile, $SRC_DIR, $covFile);
   
   if ( -z "$covFile" ) {
      `echo "EMPTY" >> $covFile 2>&1`;
   }
   print "$covFile";
}

my $configP = prepare($config);
my $startT = buildSUT();
runTests($configP, $startT);
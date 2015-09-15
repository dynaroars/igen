#!/usr/bin/perl

use strict;
use warnings;
use Time::HiRes;
use Math::BigFloat;
Math::BigFloat->precision(-3);
use util qw/harvestLCovFile/;

my $SUT_DIR="/fs/buzz/ukoc/mysql";
my $repitation=1;
my $config=$ARGV[0]; # "ssl --enable-ssl, proxy --enable-proxy, dav --disable-dav";
my $confSpaceFile; my $configListFile; my $zipFile; my $extracted;
my $ARC_DIR; my $SRC_DIR;

sub prepVars(){
   my $index= rindex $SUT_DIR, "/";
   my $SUT=substr $SUT_DIR, $index+1;
   
   opendir(DIR, $SUT_DIR) or die $!;
   while (my $file = readdir(DIR)) {
      if($file =~ /.tar/){ $zipFile="$file";
      }elsif($file =~ /config_list/){ $configListFile = "$file";
      }elsif($file =~ /config_space/){ $confSpaceFile = "$file"; }
   }
   $extracted=substr $zipFile, 0, -4;
   $SRC_DIR = "$SUT_DIR/$extracted";
   
   #`rm -fr $SRC_DIR`;
   $ARC_DIR="$SUT_DIR/archive";
   #`cd $SUT_DIR && mkdir archive`;
   `cd $SUT_DIR && tar xvf $zipFile`;
}

sub prebConfigs(){
   my $configStr="";
   my @options = split(", ",$config);
   foreach my $opt (@options){
      chomp($opt);
      my @configSettings = split(" ",$opt);
      next if($configSettings[1] =~ /NA/);
      $configStr = $configStr." ".$configSettings[1];
   }
   buildSUT($configStr);
}

sub buildSUT($){
   my $config=shift;
   my $coverageStat="$ARC_DIR/coverageStatistic.txt";
   
   # Configure
   my $start = Time::HiRes::time();
   my $logFile = "$ARC_DIR/build.$start.log";
   # Configure & Compile
   `cd $SRC_DIR && ./BUILD/compile-pentium64-gcov >> $logFile 2>&1`;
   my $finish = Time::HiRes::time();
   my $time = Math::BigFloat->new($finish - $start);
   `echo "\n---Installation Completed Finish_Time:$finish ---Total_Time:$time\n" >> $logFile 2>&1`;
   
   my $testLogFile="$ARC_DIR/test.$start.log";
   `cd $SRC_DIR && echo "---Testing:$config,Start_Time:$start\n" >> $testLogFile 2>&1`;
   `cd $SRC_DIR/mysql-test && perl mysql-test-run.pl --force --max-test-fail=10000 --gcov --retry=0 $config >> $testLogFile 2>&1`;
   $finish = Time::HiRes::time();
   $time = Math::BigFloat->new($finish - $start);
   `echo "\n---Testing Completed Finish_Time:$finish ---Total_Time:$time\n" >> $testLogFile 2>&1`;
   
   # Harvesting coverage information
   my $covFile = "$ARC_DIR/cov.$start.txt";
   my $aggregatedCovFile = "$SUT_DIR/aggregated.httpd.cov";
   `cd $SUT_DIR && lcov -c --ignore-errors='gcov,source' -d $SRC_DIR -o $aggregatedCovFile > cov.log 2>&1`;
   
   util::harvestLCovFile($aggregatedCovFile, $confSpaceFile, $SRC_DIR, $covFile);
   if ( -z "$covFile" ) { 
       `echo "EMPTY" >> $covFile 2>&1`;
   }
   print $covFile;
   `cd $SUT_DIR && rm -fr $extracted $ARC_DIR/*.log $aggregatedCovFile >> $testLogFile 2>&1`;
}

prepVars();
prebConfigs();

#!/usr/bin/perl

use strict;
use warnings;
use Time::HiRes;
use Math::BigFloat;
Math::BigFloat->precision(-3);
use util qw/harvestLCovFile/;

my $SUT_DIR="/fs/buzz/ukoc/httpd";
my $repitation=1;
my $config=$ARGV[0]; #"ssl --enable-ssl, proxy --enable-proxy, dav --disable-dav";
my $confSpaceFile; my $configListFile; my $zipFile;
my $ARC_DIR; my $BUILD_DIR; my $SRC_DIR;

sub prepVars(){
   my $index= rindex $SUT_DIR, "/";
   my $SUT=substr $SUT_DIR, $index+1;
   
   opendir(DIR, $SUT_DIR) or die $!;
   while (my $file = readdir(DIR)) {
      if($file =~ /.tar/){ $zipFile="$file";
      }elsif($file =~ /config_list/){ $configListFile = "$file";
      }elsif($file =~ /config_space/){ $confSpaceFile = "$file"; }
   }
   my $extracted=substr $zipFile, 0, -4;
   $SRC_DIR = "$SUT_DIR/$extracted";
   
   `rm -f -r $SUT_DIR/current $SRC_DIR`;
   
   `cd $SUT_DIR && mkdir current`;
   $BUILD_DIR="$SUT_DIR/current";
   
   $ARC_DIR="$SUT_DIR/archive";
   
   `cd $SUT_DIR && tar xvf $zipFile`;
}

sub prebConfigs(){
   my $configStr="";
   my @options = split(", ",$config);
   foreach my $opt (@options){
      my @configSettings = split(" ",$opt);
      $configStr = $configStr." ".$configSettings[1];
   }
   buildSUT($configStr);
}

sub buildSUT($){
   my $config=shift;
   my $coverageStat="$BUILD_DIR/coverageStatistic.txt";
   
   # Configure
   my $start = Time::HiRes::time();
   my $logFile = "$ARC_DIR/build.$start.log";
   `echo "---Configuration:$config,Start_Time:$start\n" >> $logFile 2>&1`;
   my $cov="CFLAGS='--coverage' CCFLAGS='--coverage' LDFLAGS='-lgcov' CCPFLAGS='--coverage'";
   `cd $SRC_DIR && $cov ./configure --prefix=$BUILD_DIR $config >> $logFile 2>&1`;
   `echo "\n---Configuration Completed, Continuing with make!!!\n" >> $logFile 2>&1`;
   
   # Make
   `cd $SRC_DIR && make >> $logFile 2>&1`;
   `echo "\n---Make Completed, Continuing with make install!!!\n" >> $logFile 2>&1`;
   
   # Make install
   `cd $SRC_DIR && make install >> $logFile 2>&1`;
   my $finish = Time::HiRes::time();
   my $time = Math::BigFloat->new($finish - $start);
   `echo "\n---Installation Completed Finish_Time:$finish ---Total_Time:$time\n" >> $logFile 2>&1`;
   
   #running tests
   runTests($config, $start);
   
   #Harvesting coverage information
   my $covFile = "$ARC_DIR/cov.$start.txt";
   my $aggregatedCovFile = "$SRC_DIR/aggregated.httpd.cov";
   `cd $SUT_DIR && lcov -c --ignore-errors='gcov,source' -d $SRC_DIR -o $aggregatedCovFile >> cov.log 2>&1`;
   util::harvestLCovFile($aggregatedCovFile, $confSpaceFile, $SRC_DIR, $covFile);
   
   
   # Move bins to archive
   #`mv $BUILD_DIR $ARC_DIR/$finish`;
   if ( -z "$covFile" ) { 
       `echo "EMPTY" >> $covFile 2>&1`;
   }
   print "$covFile";
}

# !!! runTest subroutine should be customized for SUT !!!
sub runTests($$){
   my $config = shift;
   my $time = shift;
   my $apxsFile = "$BUILD_DIR/bin/apxs";
   # First we need to change httpd-build/bin/apxs so that the tests can work with --coverage
   {
      if (-e $apxsFile){
         my $apsFileContent = "";
         {
            $/ = undef;
            open(IN, "< $apxsFile");
            $apsFileContent = <IN>;
            close(IN);
            $/ = "\n";
         }
         $apsFileContent =~ s/my\s+\$CFG\_CC\s+=\s+get\_vars\(\"CC\"\)\;/my \$CFG\_CC= \"gcc --coverage\"\;/;
         
         open(OUT, "> $apxsFile");
         print OUT "$apsFileContent\n";
         close(OUT);
      }
   }
   my $testLogFile="$ARC_DIR/test.$time.log";
   my $TEST_DIR="$SUT_DIR/mod_perl-2.0.8";
   
   open TESTLOG, ">>", "$testLogFile" or die $!;
   close TESTLOG;
   
   # mod_perl make and test run
   `cd $TEST_DIR && perl Makefile.PL MP_APXS=$apxsFile MP_APR_CONFIG=/usr/local/bin/apr-1-config >> $testLogFile 2>&1`;
   `cd $TEST_DIR && make && make test >> $testLogFile 2>&1`;
   `cd $TEST_DIR && make clean >> $testLogFile 2>&1`;
}

prepVars();
prebConfigs();

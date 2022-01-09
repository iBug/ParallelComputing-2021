#!/usr/bin/perl

use strict;
use 5.010;
use experimental qw/switch/;

die "Need an action\n" unless @ARGV > 0;

sub random {
    return int rand $_[0];
}

given (shift(@ARGV)) {
    when (/^g/) {
        my $n = int $ARGV[0];
        my $max;

        die "Need a positive integer\n" unless $n > 0;

        if (@ARGV > 1) {
            $max = int $ARGV[1];
        } else {
            $max = 2 * $n;
        }
        say $n;
        say random $max for 1 .. $n;
    }
    when (/^c/) {
        my $last = -1;

        while (<>) {
            chomp;
            last if /^$/;
            die "Wrong value ($last > $_) at line $.\n" if $last > $_;
            $last = $_;
        }
        say 'OK';
    }
}
